import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac_utils import soft_update, hard_update
from sac_model import QNetwork
from sac_model import CirclePF
from sac_sampling import sample_actions


class SAC(object):
    def __init__(self, args, env):

        self.gamma = 1.0  # Fixed to 1.0 for GFlowNet
        self.target_update_interval = args.target_update_interval
        self.device = env.device
        self.tau = args.tau
        self.env = env  # Store env for sample_actions
        self.critic = QNetwork(env.dim, env.dim, args.Critic_hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = QNetwork(env.dim, env.dim, args.Critic_hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.policy = CirclePF(
            hidden_dim=args.hidden_dim,
            n_hidden=args.n_hidden,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
        ).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        # Add scheduler for policy optimizer
        self.policy_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.policy_optim,
            milestones=[i * args.scheduler_milestone for i in range(1, 10)],
            gamma=args.gamma_scheduler,
        )

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size=batch_size)

        # ★ 여기에서 그래프를 확실히 끊어주기
        state_batch      = state_batch.detach().to(self.device)
        action_batch     = action_batch.detach().to(self.device)
        reward_batch     = reward_batch.detach().to(self.device)
        next_state_batch = next_state_batch.detach().to(self.device)
        done_batch       = done_batch.detach().to(self.device)

        with torch.no_grad():
            # Separate next_state_batch based on done flag
            done_mask = (done_batch == 1).squeeze(-1)  # (batch_size,)
            # print(done_mask)
            next_state_not_done = next_state_batch[~done_mask]  # States where done == 0
            # print(next_state_not_done)
            target_q_value = reward_batch[~done_mask].squeeze(-1)
            # print(target_q_value)           

            _, _, next_state_exit_proba, next_state_action_naive, next_state_log_pi_naive = sample_actions(self.env, self.policy, next_state_not_done)

            is_inf_mask = torch.all(torch.isinf(next_state_action_naive), dim=-1)
        #     # 1. 반드시 terminal state에 도달하는 경우

            target_q_value += next_state_exit_proba * (self.env.reward(next_state_not_done) - next_state_exit_proba.log())
        #     # 2. 반드시 terminal state에 도달하지 않아도 되는 경우 

            qf1_next_target, qf2_next_target = self.critic_target(next_state_not_done[~is_inf_mask], next_state_action_naive[~is_inf_mask])
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target).squeeze(-1) 

            target_q_value[~is_inf_mask] += (1 - next_state_exit_proba[~is_inf_mask]) * (min_qf_next_target - next_state_log_pi_naive[~is_inf_mask])

        qf1, qf2 = self.critic(state_batch[~done_mask], action_batch[~done_mask])  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1 = qf1.squeeze(-1)
        qf2 = qf2.squeeze(-1)
        qf1_loss = F.mse_loss(qf1, target_q_value)  
        qf2_loss = F.mse_loss(qf2, target_q_value)  

        qf_loss = qf1_loss + qf2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        s0_mask = torch.all(state_batch == 0, dim=-1)
        non_s0_mask = ~s0_mask

        policy_loss = torch.zeros_like(reward_batch).squeeze(-1)

        if s0_mask.any():
            _, _, exit_proba_s0, action_naive_s0, log_pi_naive_s0 = sample_actions(self.env, self.policy, state_batch[s0_mask])

        if non_s0_mask.any():
            _, _, exit_proba_non_s0, action_naive_non_s0, log_pi_naive_non_s0 = sample_actions(self.env, self.policy, state_batch[non_s0_mask])

        exit_proba = torch.cat([exit_proba_s0, exit_proba_non_s0], dim=0)
        action_naive = torch.cat([action_naive_s0, action_naive_non_s0], dim=0)
        log_pi_naive = torch.cat([log_pi_naive_s0, log_pi_naive_non_s0], dim=0)
        state_batch_reordered = torch.cat([state_batch[s0_mask], state_batch[non_s0_mask]], dim=0)
        is_inf_mask = torch.all(torch.isinf(action_naive), dim=-1)
        is_non_zero_mask = torch.where(exit_proba != 0, True, False)

        policy_loss[is_non_zero_mask] += exit_proba[is_non_zero_mask] * (exit_proba[is_non_zero_mask].log() - self.env.reward(state_batch_reordered[is_non_zero_mask]))


        qf1_pi, qf2_pi = self.critic(state_batch_reordered[~is_inf_mask], action_naive[~is_inf_mask])

        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        min_qf_pi = min_qf_pi.squeeze(-1)
        policy_loss[~is_inf_mask] += (1 - exit_proba[~is_inf_mask]) * (log_pi_naive[~is_inf_mask] - min_qf_pi)
        policy_loss = policy_loss.mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item()
        
    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
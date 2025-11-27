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
            # Problem 1: Handle sink states in next_state_batch
            # Identify non-sink next states
            non_sink_mask = ~torch.all(next_state_batch == self.env.sink_state, dim=-1)

            # Initialize with zeros (will be masked out anyway)
            next_state_action = torch.zeros_like(next_state_batch)
            next_state_log_pi = torch.zeros(next_state_batch.shape[0], device=self.device)

            # Only sample actions for non-sink next states
            if non_sink_mask.any():
                sampled_actions, sampled_log_pi = sample_actions(
                    self.env, self.policy, next_state_batch[non_sink_mask]
                )
                # Replace -inf terminal actions with zeros (won't be used anyway due to done_batch)
                sampled_actions = torch.where(  
                    torch.isinf(sampled_actions),
                    torch.zeros_like(sampled_actions),
                    sampled_actions
                )
                next_state_action[non_sink_mask] = sampled_actions
                next_state_log_pi[non_sink_mask] = sampled_log_pi

            # Replace sink states with zeros for critic input (won't be used due to done_batch)
            next_state_batch_clean = torch.where(
                torch.isinf(next_state_batch),
                torch.zeros_like(next_state_batch),
                next_state_batch
            )

            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch_clean, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - next_state_log_pi.unsqueeze(-1)
            # FIXED: done_batch is "done" (1=done, 0=not done)
            # Correct SAC Bellman: Q(s,a) = r + (1-done) * γ * Q(s',a')
            # Intermediate (done=0): Q = r + 1*γ*Q' = r + γ*Q' ✓
            # Terminal (done=1): Q = r + 0*γ*Q' = r ✓
            next_q_value = reward_batch + (1 - done_batch) * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        s0_mask = torch.all(state_batch == 0, dim=-1)
        non_s0_mask = ~s0_mask
        if s0_mask.any():
            pi_s0, log_pi_s0 = sample_actions(self.env, self.policy, state_batch[s0_mask])

        if non_s0_mask.any():
            pi_non_s0, log_pi_non_s0 = sample_actions(self.env, self.policy, state_batch[non_s0_mask])
            
        pi = torch.cat([pi_s0, pi_non_s0], dim=0)
        log_pi = torch.cat([log_pi_s0, log_pi_non_s0], dim=0)
        state_batch_reordered = torch.cat([state_batch[s0_mask], state_batch[non_s0_mask]], dim=0)

        qf1_pi, qf2_pi = self.critic(state_batch_reordered, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (log_pi - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

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
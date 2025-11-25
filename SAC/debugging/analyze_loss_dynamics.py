"""
Analyze why policy loss keeps increasing by tracking Q-values, entropy, and other metrics
"""
import torch
import numpy as np
from env import Box, get_last_states
from sac import SAC
from model import CirclePB
from sampling import sample_trajectories, evaluate_backward_logprobs, sample_actions
from sac_replay_memory import ReplayMemory, trajectories_to_transitions
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--dim", type=int, default=2)
parser.add_argument("--delta", type=float, default=0.25)
parser.add_argument("--env_epsilon", type=float, default=1e-10)
parser.add_argument("--n_components", type=int, default=2)
parser.add_argument("--n_components_s0", type=int, default=4)
parser.add_argument("--beta_min", type=float, default=0.1)
parser.add_argument("--beta_max", type=float, default=5.0)
parser.add_argument("--PB", type=str, default="uniform")
parser.add_argument("--gamma_scheduler", type=float, default=0.5)
parser.add_argument("--scheduler_milestone", type=int, default=2500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--BS", type=int, default=128)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--n_hidden", type=int, default=3)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--sac_alpha", type=float, default=0.2)
parser.add_argument("--automatic_entropy_tuning", action="store_true", default=False)
parser.add_argument("--target_update_interval", type=int, default=1)
parser.add_argument("--Critic_hidden_size", type=int, default=256)
parser.add_argument("--replay_size", type=int, default=1000000)
parser.add_argument("--sac_batch_size", type=int, default=256)
parser.add_argument("--updates_per_step", type=int, default=1)
parser.add_argument("--reward_type", type=str, default="baseline")
parser.add_argument("--reward_debug", action="store_true", default=False)

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = args.device
env = Box(dim=args.dim, delta=args.delta, epsilon=args.env_epsilon,
          device_str=device, reward_type=args.reward_type, reward_debug=args.reward_debug)

sac_agent = SAC(args, env)
bw_model = CirclePB(hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,
                     torso=sac_agent.policy.torso if args.PB == "tied" else None,
                     uniform=args.PB == "uniform", n_components=args.n_components,
                     beta_min=args.beta_min, beta_max=args.beta_max).to(device)

memory = ReplayMemory(args.replay_size, args.seed, device=device)

print("="*80)
print("COLLECTING INITIAL DATA")
print("="*80)

# Collect initial data
for i in range(20):
    trajectories, actionss, logprobs, all_logprobs = sample_trajectories(env, sac_agent.policy, args.BS)
    last_states = get_last_states(env, trajectories)
    logrewards = env.reward(last_states).log()
    bw_logprobs, all_bw_logprobs = evaluate_backward_logprobs(env, bw_model, trajectories)
    all_states, all_actions, all_rewards, all_next_states, all_dones = trajectories_to_transitions(
        trajectories, actionss, all_bw_logprobs, last_states, logrewards, env
    )
    memory.push_batch(all_states, all_actions, all_rewards, all_next_states, all_dones)

print(f"Collected {len(memory)} transitions")

print("\n" + "="*80)
print("TRAINING AND TRACKING DYNAMICS")
print("="*80)

sac_updates = 0
for step in range(50):
    qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha = sac_agent.update_parameters(
        memory, args.sac_batch_size, sac_updates
    )
    sac_updates += 1

    if step % 10 == 0:
        # Sample batch and analyze
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(args.sac_batch_size)

        with torch.no_grad():
            # Current Q-values for replay buffer actions
            qf1, qf2 = sac_agent.critic(state_batch, action_batch)
            min_q_replay = torch.min(qf1, qf2)

            # Sample new actions from policy
            s0_mask = torch.all(state_batch == 0, dim=-1)
            non_s0_mask = ~s0_mask

            pi = torch.zeros_like(state_batch)
            log_pi = torch.zeros(state_batch.shape[0], device=device)

            if s0_mask.any():
                pi_s0, log_pi_s0 = sample_actions(env, sac_agent.policy, state_batch[s0_mask])
                pi[s0_mask] = pi_s0
                log_pi[s0_mask] = log_pi_s0

            if non_s0_mask.any():
                pi_non_s0, log_pi_non_s0 = sample_actions(env, sac_agent.policy, state_batch[non_s0_mask])
                pi[non_s0_mask] = pi_non_s0
                log_pi[non_s0_mask] = log_pi_non_s0

            # Check terminal actions
            is_terminal = torch.isinf(pi).any(dim=-1)
            valid_mask = ~is_terminal

            if valid_mask.any():
                valid_pi = pi[valid_mask]
                valid_pi = torch.where(torch.isinf(valid_pi), torch.zeros_like(valid_pi), valid_pi)
                valid_states = state_batch[valid_mask]
                valid_log_pi = log_pi[valid_mask]

                qf1_pi, qf2_pi = sac_agent.critic(valid_states, valid_pi)
                min_q_policy = torch.min(qf1_pi, qf2_pi)

                # Policy loss components
                entropy_term = (args.sac_alpha * valid_log_pi).mean()
                q_term = min_q_policy.mean()

                print(f"\nStep {step}:")
                print(f"  Policy Loss: {policy_loss:.4f}")
                print(f"  Critic Loss: Q1={qf1_loss:.4f}, Q2={qf2_loss:.4f}")
                print(f"  Q-values (replay): mean={min_q_replay.mean():.4f}, std={min_q_replay.std():.4f}")
                print(f"  Q-values (policy): mean={q_term:.4f}, std={min_q_policy.std():.4f}")
                print(f"  Entropy term (α*log_π): {entropy_term:.4f}")
                print(f"  Rewards: mean={reward_batch.mean():.4f}, std={reward_batch.std():.4f}")
                print(f"  Terminal actions: {is_terminal.sum().item()}/{len(is_terminal)}")
                print(f"  Learning rate: {sac_agent.policy_optim.param_groups[0]['lr']:.6f}")

                # Check if Q-values are decreasing
                if step > 0 and q_term < prev_q_term:
                    print(f"  ⚠️  Q-values DECREASED: {prev_q_term:.4f} -> {q_term:.4f}")
                prev_q_term = q_term
            else:
                print(f"\nStep {step}: ALL ACTIONS ARE TERMINAL!")
                prev_q_term = 0

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print("\nPolicy loss = α * log_π - Q")
print("If policy loss increases:")
print("  1. log_π is decreasing (policy becomes more deterministic)")
print("  2. Q is decreasing (value estimates are getting worse)")
print("\nPossible causes:")
print("  - Q-values are not learning correctly")
print("  - Reward structure creates negative cumulative rewards")
print("  - Policy is collapsing to avoid exploration")
print("  - Target network updates are too slow")

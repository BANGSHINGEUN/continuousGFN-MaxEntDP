"""
Debug script to understand why policy loss is increasing
"""
import torch
import numpy as np
from sac import SAC
from env import Box
from sac_replay_memory import ReplayMemory, trajectories_to_transitions
from sampling import sample_trajectories, evaluate_backward_logprobs
from model import CirclePB
import argparse

# Setup
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
print(f"Using device: {device}")

# Create environment
env = Box(
    dim=args.dim,
    delta=args.delta,
    epsilon=args.env_epsilon,
    device_str=device,
    reward_type=args.reward_type,
    reward_debug=args.reward_debug,
)

# Create SAC agent
sac_agent = SAC(args, env)

# Create replay memory
memory = ReplayMemory(args.replay_size, args.seed, device=device)

# Create backward model
bw_model = CirclePB(
    hidden_dim=args.hidden_dim,
    n_hidden=args.n_hidden,
    torso=sac_agent.policy.torso if args.PB == "tied" else None,
    uniform=args.PB == "uniform",
    n_components=args.n_components,
    beta_min=args.beta_min,
    beta_max=args.beta_max,
).to(device)

print("\n" + "="*80)
print("COLLECTING INITIAL DATA")
print("="*80)

# Collect some initial data
for i in range(5):
    trajectories, actionss, logprobs, all_logprobs = sample_trajectories(
        env, sac_agent.policy, args.BS
    )
    last_states = torch.stack([traj[torch.all(traj != env.sink_state, dim=-1)][-1]
                                for traj in trajectories])
    logrewards = env.reward(last_states).log()
    bw_logprobs, all_bw_logprobs = evaluate_backward_logprobs(
        env, bw_model, trajectories
    )

    all_states, all_actions, all_rewards, all_next_states, all_dones = trajectories_to_transitions(
        trajectories, actionss, all_bw_logprobs, last_states, logrewards, env
    )
    memory.push_batch(all_states, all_actions, all_rewards, all_next_states, all_dones)

print(f"Collected {len(memory)} transitions")

# Sample a batch and analyze
print("\n" + "="*80)
print("ANALYZING REPLAY BUFFER STATISTICS")
print("="*80)

state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(args.sac_batch_size)

print(f"State batch shape: {state_batch.shape}")
print(f"Action batch shape: {action_batch.shape}")
print(f"Reward batch stats: min={reward_batch.min().item():.4f}, max={reward_batch.max().item():.4f}, mean={reward_batch.mean().item():.4f}")
print(f"Mask batch (done) stats: mean={mask_batch.mean().item():.4f} (1=not done, 0=done)")
print(f"Next state batch - sink states: {torch.all(next_state_batch == env.sink_state, dim=-1).sum().item()}/{len(next_state_batch)}")

# Analyze initial Q-values
print("\n" + "="*80)
print("INITIAL Q-VALUES")
print("="*80)

with torch.no_grad():
    qf1, qf2 = sac_agent.critic(state_batch, action_batch)
    print(f"Q1 stats: min={qf1.min().item():.4f}, max={qf1.max().item():.4f}, mean={qf1.mean().item():.4f}")
    print(f"Q2 stats: min={qf2.min().item():.4f}, max={qf2.max().item():.4f}, mean={qf2.mean().item():.4f}")

# Train for a few steps and track metrics
print("\n" + "="*80)
print("TRAINING AND TRACKING METRICS")
print("="*80)

sac_updates = 0
for step in range(10):
    qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha = sac_agent.update_parameters(
        memory, args.sac_batch_size, sac_updates
    )
    sac_updates += 1

    # Get detailed metrics
    state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(args.sac_batch_size)

    with torch.no_grad():
        # Current Q-values
        qf1, qf2 = sac_agent.critic(state_batch, action_batch)
        min_q = torch.min(qf1, qf2)

        # Policy's Q-values
        from sampling import sample_actions
        pi, log_pi = sample_actions(env, sac_agent.policy, state_batch)
        pi = torch.where(torch.isinf(pi), torch.zeros_like(pi), pi)
        qf1_pi, qf2_pi = sac_agent.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Policy loss components
        entropy_term = alpha * log_pi.mean()
        q_term = min_qf_pi.mean()

    print(f"Step {step+1}:")
    print(f"  Critic loss: Q1={qf1_loss:.4f}, Q2={qf2_loss:.4f}")
    print(f"  Policy loss: {policy_loss:.4f}")
    print(f"    - Alpha * log_pi: {entropy_term:.4f}")
    print(f"    - Q(s, pi(s)): {q_term:.4f}")
    print(f"  Alpha: {alpha:.4f}")
    print(f"  Current LR: {sac_agent.policy_optim.param_groups[0]['lr']:.6f}")

    # Check for issues
    if torch.isnan(torch.tensor(policy_loss)):
        print("  WARNING: NaN in policy loss!")
        break
    if q_term < -10:
        print("  WARNING: Very negative Q-values!")
    if entropy_term < -10:
        print("  WARNING: Very negative entropy term!")

print("\n" + "="*80)
print("FINAL Q-VALUE ANALYSIS")
print("="*80)

with torch.no_grad():
    qf1, qf2 = sac_agent.critic(state_batch, action_batch)
    print(f"Final Q1 stats: min={qf1.min().item():.4f}, max={qf1.max().item():.4f}, mean={qf1.mean().item():.4f}")
    print(f"Final Q2 stats: min={qf2.min().item():.4f}, max={qf2.max().item():.4f}, mean={qf2.mean().item():.4f}")

print("\nDone!")

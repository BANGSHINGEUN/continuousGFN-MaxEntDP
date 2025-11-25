"""
Check if terminal actions are being sampled during policy updates
"""
import torch
import numpy as np
from env import Box
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

# Collect some data
print("Collecting data...")
for i in range(10):
    trajectories, actionss, logprobs, all_logprobs = sample_trajectories(env, sac_agent.policy, args.BS)
    from env import get_last_states
    last_states = get_last_states(env, trajectories)
    logrewards = env.reward(last_states).log()
    bw_logprobs, all_bw_logprobs = evaluate_backward_logprobs(env, bw_model, trajectories)
    all_states, all_actions, all_rewards, all_next_states, all_dones = trajectories_to_transitions(
        trajectories, actionss, all_bw_logprobs, last_states, logrewards, env
    )
    memory.push_batch(all_states, all_actions, all_rewards, all_next_states, all_dones)

print(f"Collected {len(memory)} transitions")

# Sample a batch
state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(256)

print("\n" + "="*80)
print("CHECKING TERMINAL ACTIONS IN POLICY LOSS")
print("="*80)

# Check states near boundary
near_boundary = torch.any(state_batch >= 1 - env.delta - 0.1, dim=-1)
print(f"\nStates near boundary: {near_boundary.sum().item()}/{len(state_batch)}")

if near_boundary.any():
    print("\nSampling actions from policy for boundary states:")
    boundary_states = state_batch[near_boundary]
    print(f"Boundary states shape: {boundary_states.shape}")
    print(f"First 3 boundary states:\n{boundary_states[:3]}")

    pi, log_pi = sample_actions(env, sac_agent.policy, boundary_states)

    print(f"\nSampled actions shape: {pi.shape}")
    print(f"First 3 actions:\n{pi[:3]}")

    # Check for -inf (terminal actions)
    is_terminal = torch.isinf(pi)
    print(f"\n Terminal actions: {is_terminal.any(dim=-1).sum().item()}/{len(pi)}")

    if is_terminal.any(dim=-1).any():
        print("\n⚠️ PROBLEM: Terminal actions (-inf) are being sampled!")
        print("These are currently replaced with zeros, which is WRONG!")
        print("\nWhat happens:")
        print("1. Policy samples terminal action [-inf, -inf]")
        print("2. Code replaces it with [0, 0]")
        print("3. Q([0, 0]) is computed and used for policy gradient")
        print("4. Policy learns wrong signal!")

print("\n" + "="*80)
print("SOLUTION")
print("="*80)

print("\nWe should:")
print("1. Detect terminal actions (inf values)")
print("2. EXCLUDE these samples from policy loss computation")
print("3. Or handle terminal actions separately")
print("\nTerminal actions should NOT be replaced with zeros!")

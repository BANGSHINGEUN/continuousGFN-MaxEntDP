"""
Test code for SAC components
Tests: trajectories_to_transitions, ReplayMemory, SAC initialization
"""
import torch
import numpy as np
from env import Box
from model import CirclePF
from sampling import sample_trajectories, evaluate_backward_logprobs
from sac_replay_memory import ReplayMemory, trajectories_to_transitions
from sac import SAC
import argparse

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create simple environment
env = Box(dim=2, delta=0.1, epsilon=0.0, device_str=device, reward_type="baseline")

# Create minimal args for SAC
args = argparse.Namespace(
    device=device,
    dim=2,
    delta=0.1,
    hidden_dim=64,
    n_hidden=2,
    n_components=4,
    n_components_s0=2,
    beta_min=1e-2,
    beta_max=1e2,
    lr=3e-4,
    tau=0.005,
    sac_alpha=0.2,
    automatic_entropy_tuning=False,
    target_update_interval=1,
    Critic_hidden_size=256,
    gamma_scheduler=0.99,
    scheduler_milestone=10000,
    replay_size=10000,
    sac_batch_size=64,
)

print("\n" + "="*60)
print("Test 1: CirclePF Policy Initialization")
print("="*60)

policy = CirclePF(
    hidden_dim=args.hidden_dim,
    n_hidden=args.n_hidden,
    n_components=args.n_components,
    n_components_s0=args.n_components_s0,
    beta_min=args.beta_min,
    beta_max=args.beta_max,
).to(device)

print(f"✓ Policy created successfully")
print(f"  Policy parameters: {sum(p.numel() for p in policy.parameters())}")

print("\n" + "="*60)
print("Test 2: Sample Trajectories")
print("="*60)

BS = 8
trajectories, actionss, logprobs, all_logprobs = sample_trajectories(env, policy, BS)

print(f"✓ Sampled {BS} trajectories")
print(f"  Trajectories shape: {trajectories.shape}")
print(f"  Actions shape: {actionss.shape}")
print(f"  Logprobs shape: {logprobs.shape}")
print(f"  All logprobs shape: {all_logprobs.shape}")
print(f"  Device: {trajectories.device}")

print("\n" + "="*60)
print("Test 3: trajectories_to_transitions Function")
print("="*60)

from env import get_last_states
from model import CirclePB

bw_model = CirclePB(
    hidden_dim=args.hidden_dim,
    n_hidden=args.n_hidden,
    uniform=False,
    n_components=args.n_components,
    beta_min=args.beta_min,
    beta_max=args.beta_max,
).to(device)

last_states = get_last_states(env, trajectories)
logrewards = env.reward(last_states).log()
bw_logprobs, all_bw_logprobs = evaluate_backward_logprobs(env, bw_model, trajectories)

print(f"  Last states shape: {last_states.shape}")
print(f"  Logrewards shape: {logrewards.shape}")
print(f"  All bw_logprobs shape: {all_bw_logprobs.shape}")

all_states, all_actions, all_rewards, all_next_states, all_dones = trajectories_to_transitions(
    trajectories, actionss, all_bw_logprobs, last_states, logrewards, env
)

print(f"✓ Converted trajectories to transitions")
print(f"  Total transitions: {all_states.shape[0]}")
print(f"  States shape: {all_states.shape}")
print(f"  Actions shape: {all_actions.shape}")
print(f"  Rewards shape: {all_rewards.shape}")
print(f"  Next states shape: {all_next_states.shape}")
print(f"  Dones shape: {all_dones.shape}")
print(f"  Device: {all_states.device}")

# Check for NaNs
print(f"\n  Checking for NaNs:")
print(f"    States: {torch.isnan(all_states).any().item()}")
print(f"    Actions: {torch.isnan(all_actions).any().item()}")
print(f"    Rewards: {torch.isnan(all_rewards).any().item()}")
print(f"    Next states: {torch.isnan(all_next_states).any().item()}")
print(f"    Dones: {torch.isnan(all_dones).any().item()}")

# Check terminal transitions
num_terminal = all_dones.sum().item()
print(f"\n  Terminal transitions: {num_terminal} / {all_states.shape[0]}")
print(f"  Intermediate transitions: {all_states.shape[0] - num_terminal}")

# Verify sink states in next_states for terminal transitions
terminal_mask = all_dones == 1.0
terminal_next_states = all_next_states[terminal_mask]
is_sink = torch.all(terminal_next_states == env.sink_state, dim=-1)
print(f"  All terminal next_states are sink: {is_sink.all().item()}")

print("\n" + "="*60)
print("Test 4: ReplayMemory")
print("="*60)

memory = ReplayMemory(args.replay_size, seed=42, device=device)
print(f"✓ ReplayMemory created with capacity {args.replay_size}")
print(f"  Initial size: {len(memory)}")

# Push batch
memory.push_batch(all_states, all_actions, all_rewards, all_next_states, all_dones)
print(f"✓ Pushed {all_states.shape[0]} transitions")
print(f"  Current size: {len(memory)}")

# Sample batch
if len(memory) >= args.sac_batch_size:
    sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = memory.sample(args.sac_batch_size)
    print(f"✓ Sampled batch of {args.sac_batch_size}")
    print(f"  Sampled states shape: {sampled_states.shape}")
    print(f"  Sampled states device: {sampled_states.device}")
    print(f"  All sampled data on {device}: {sampled_states.device == torch.device(device)}")
else:
    print(f"  Not enough samples to test sampling (need {args.sac_batch_size}, have {len(memory)})")

# Test wrap-around by filling buffer
print(f"\n  Testing buffer wrap-around:")
for i in range(5):
    trajectories_new, actionss_new, _, _ = sample_trajectories(env, policy, BS)
    last_states_new = get_last_states(env, trajectories_new)
    logrewards_new = env.reward(last_states_new).log()
    _, all_bw_logprobs_new = evaluate_backward_logprobs(env, bw_model, trajectories_new)

    states_new, actions_new, rewards_new, next_states_new, dones_new = trajectories_to_transitions(
        trajectories_new, actionss_new, all_bw_logprobs_new, last_states_new, logrewards_new, env
    )
    memory.push_batch(states_new, actions_new, rewards_new, next_states_new, dones_new)

print(f"  After {5} more batches, size: {len(memory)}")

print("\n" + "="*60)
print("Test 5: SAC Initialization")
print("="*60)

sac_agent = SAC(args, env)
print(f"✓ SAC agent created")
print(f"  Gamma (should be 1.0): {sac_agent.gamma}")
print(f"  Alpha: {sac_agent.alpha}")
print(f"  Tau: {sac_agent.tau}")
print(f"  Device: {sac_agent.device}")
print(f"  Policy parameters: {sum(p.numel() for p in sac_agent.policy.parameters())}")
print(f"  Critic parameters: {sum(p.numel() for p in sac_agent.critic.parameters())}")

print("\n" + "="*60)
print("Test 6: SAC Update")
print("="*60)

if len(memory) >= args.sac_batch_size:
    try:
        qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha = sac_agent.update_parameters(
            memory, args.sac_batch_size, updates=0
        )
        print(f"✓ SAC update successful")
        print(f"  QF1 loss: {qf1_loss:.4f}")
        print(f"  QF2 loss: {qf2_loss:.4f}")
        print(f"  Policy loss: {policy_loss:.4f}")
        print(f"  Alpha loss: {alpha_loss:.4f}")
        print(f"  Alpha: {alpha:.4f}")

        # Check for NaNs in policy parameters
        has_nan = any([torch.isnan(p).any() for p in sac_agent.policy.parameters()])
        print(f"  Policy has NaN: {has_nan}")

        # Try multiple updates
        print(f"\n  Testing multiple updates:")
        for update_i in range(5):
            qf1, qf2, pol, alp_loss, alp = sac_agent.update_parameters(
                memory, args.sac_batch_size, updates=update_i+1
            )
            print(f"    Update {update_i+1}: QF1={qf1:.4f}, QF2={qf2:.4f}, Pol={pol:.4f}")

    except Exception as e:
        print(f"✗ SAC update failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"  Not enough samples for SAC update (need {args.sac_batch_size}, have {len(memory)})")

print("\n" + "="*60)
print("Test 7: End-to-End Integration")
print("="*60)

# Clear memory and do a fresh run
memory_fresh = ReplayMemory(args.replay_size, seed=42, device=device)
sac_agent_fresh = SAC(args, env)

print(f"  Running 10 iterations of sampling + updating:")
for iter_i in range(10):
    # Sample trajectories
    trajs, acts, _, _ = sample_trajectories(env, sac_agent_fresh.policy, BS)
    last_sts = get_last_states(env, trajs)
    logrews = env.reward(last_sts).log()
    _, all_bw_lps = evaluate_backward_logprobs(env, bw_model, trajs)

    # Convert and push
    sts, acts_t, rews, next_sts, dns = trajectories_to_transitions(
        trajs, acts, all_bw_lps, last_sts, logrews, env
    )
    memory_fresh.push_batch(sts, acts_t, rews, next_sts, dns)

    # Update if enough samples
    if len(memory_fresh) >= args.sac_batch_size:
        q1, q2, pol, _, _ = sac_agent_fresh.update_parameters(
            memory_fresh, args.sac_batch_size, updates=iter_i
        )
        status = f"✓ Iter {iter_i+1}: Mem={len(memory_fresh)}, Q1={q1:.3f}, Q2={q2:.3f}, Pol={pol:.3f}"
    else:
        status = f"  Iter {iter_i+1}: Mem={len(memory_fresh)} (waiting for {args.sac_batch_size})"

    print(status)

print("\n" + "="*60)
print("All Tests Completed!")
print("="*60)

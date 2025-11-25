"""
Debug NaN issue in SAC update
"""
import torch
import numpy as np
from env import Box
from model import CirclePF, CirclePB
from sampling import sample_trajectories, evaluate_backward_logprobs
from sac_replay_memory import ReplayMemory, trajectories_to_transitions
from sac import SAC
from env import get_last_states
import argparse

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create environment
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

# Create policy and backward model
policy = CirclePF(
    hidden_dim=args.hidden_dim,
    n_hidden=args.n_hidden,
    n_components=args.n_components,
    n_components_s0=args.n_components_s0,
    beta_min=args.beta_min,
    beta_max=args.beta_max,
).to(device)

bw_model = CirclePB(
    hidden_dim=args.hidden_dim,
    n_hidden=args.n_hidden,
    uniform=False,
    n_components=args.n_components,
    beta_min=args.beta_min,
    beta_max=args.beta_max,
).to(device)

# Sample trajectories and create transitions
BS = 16
trajectories, actionss, _, _ = sample_trajectories(env, policy, BS)
last_states = get_last_states(env, trajectories)
logrewards = env.reward(last_states).log()
_, all_bw_logprobs = evaluate_backward_logprobs(env, bw_model, trajectories)

print("\n" + "="*60)
print("Checking trajectory data")
print("="*60)
print(f"Trajectories shape: {trajectories.shape}")
print(f"Last states shape: {last_states.shape}")
print(f"Logrewards shape: {logrewards.shape}")
print(f"Logrewards stats: min={logrewards.min().item():.4f}, max={logrewards.max().item():.4f}, mean={logrewards.mean().item():.4f}")
print(f"Logrewards has NaN: {torch.isnan(logrewards).any().item()}")
print(f"Logrewards has Inf: {torch.isinf(logrewards).any().item()}")
print(f"All_bw_logprobs shape: {all_bw_logprobs.shape}")
print(f"All_bw_logprobs stats: min={all_bw_logprobs.min().item():.4f}, max={all_bw_logprobs.max().item():.4f}, mean={all_bw_logprobs.mean().item():.4f}")
print(f"All_bw_logprobs has NaN: {torch.isnan(all_bw_logprobs).any().item()}")
print(f"All_bw_logprobs has Inf: {torch.isinf(all_bw_logprobs).any().item()}")

# Convert to transitions
all_states, all_actions, all_rewards, all_next_states, all_dones = trajectories_to_transitions(
    trajectories, actionss, all_bw_logprobs, last_states, logrewards, env
)

print("\n" + "="*60)
print("Checking transition data")
print("="*60)
print(f"States shape: {all_states.shape}")
print(f"Rewards shape: {all_rewards.shape}")
print(f"Rewards stats: min={all_rewards.min().item():.4f}, max={all_rewards.max().item():.4f}, mean={all_rewards.mean().item():.4f}")
print(f"Rewards has NaN: {torch.isnan(all_rewards).any().item()}")
print(f"Rewards has Inf: {torch.isinf(all_rewards).any().item()}")

# Create replay memory and push data
memory = ReplayMemory(args.replay_size, seed=42, device=device)

# Fill buffer with enough samples
for i in range(5):
    trajs, acts, _, _ = sample_trajectories(env, policy, BS)
    last_sts = get_last_states(env, trajs)
    logrews = env.reward(last_sts).log()
    _, all_bw_lps = evaluate_backward_logprobs(env, bw_model, trajs)
    sts, acts_t, rews, next_sts, dns = trajectories_to_transitions(
        trajs, acts, all_bw_lps, last_sts, logrews, env
    )
    memory.push_batch(sts, acts_t, rews, next_sts, dns)

print(f"\nMemory size: {len(memory)}")

# Create SAC agent
sac_agent = SAC(args, env)

# Sample a batch and check values
state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(args.sac_batch_size)

print("\n" + "="*60)
print("Checking sampled batch")
print("="*60)
print(f"State batch shape: {state_batch.shape}")
print(f"Action batch shape: {action_batch.shape}")
print(f"Reward batch shape: {reward_batch.shape}")
print(f"Reward batch stats: min={reward_batch.min().item():.4f}, max={reward_batch.max().item():.4f}, mean={reward_batch.mean().item():.4f}")
print(f"Reward batch has NaN: {torch.isnan(reward_batch).any().item()}")
print(f"Reward batch has Inf: {torch.isinf(reward_batch).any().item()}")
print(f"Mask batch shape: {mask_batch.shape}")
print(f"Mask batch values: {mask_batch.unique()}")

# Check critic output
with torch.no_grad():
    qf1, qf2 = sac_agent.critic(state_batch, action_batch)
    print(f"\n" + "="*60)
    print("Checking Critic output")
    print("="*60)
    print(f"QF1 shape: {qf1.shape}")
    print(f"QF1 stats: min={qf1.min().item():.4f}, max={qf1.max().item():.4f}, mean={qf1.mean().item():.4f}")
    print(f"QF1 has NaN: {torch.isnan(qf1).any().item()}")
    print(f"QF1 has Inf: {torch.isinf(qf1).any().item()}")

    # Check next state action sampling
    from sampling import sample_actions

    non_sink_mask = ~torch.all(next_state_batch == env.sink_state, dim=-1)
    print(f"\n" + "="*60)
    print("Checking next state action sampling")
    print("="*60)
    print(f"Non-sink mask: {non_sink_mask.sum().item()} / {non_sink_mask.shape[0]}")

    if non_sink_mask.any():
        next_actions, next_log_pi = sample_actions(env, sac_agent.policy, next_state_batch[non_sink_mask])
        print(f"Next actions shape: {next_actions.shape}")
        print(f"Next actions has NaN: {torch.isnan(next_actions).any().item()}")
        print(f"Next actions has Inf: {torch.isinf(next_actions).any().item()}")
        print(f"Next log_pi shape: {next_log_pi.shape}")
        print(f"Next log_pi stats: min={next_log_pi.min().item():.4f}, max={next_log_pi.max().item():.4f}, mean={next_log_pi.mean().item():.4f}")
        print(f"Next log_pi has NaN: {torch.isnan(next_log_pi).any().item()}")
        print(f"Next log_pi has Inf: {torch.isinf(next_log_pi).any().item()}")

        # Check critic target output
        next_state_action = torch.zeros_like(next_state_batch)
        next_state_action[non_sink_mask] = next_actions

        qf1_next, qf2_next = sac_agent.critic_target(next_state_batch, next_state_action)
        print(f"\nQF1_next shape: {qf1_next.shape}")
        print(f"QF1_next stats: min={qf1_next.min().item():.4f}, max={qf1_next.max().item():.4f}, mean={qf1_next.mean().item():.4f}")
        print(f"QF1_next has NaN: {torch.isnan(qf1_next).any().item()}")
        print(f"QF1_next has Inf: {torch.isinf(qf1_next).any().item()}")

print("\n" + "="*60)
print("Running full SAC update with detailed logging")
print("="*60)

# Add hooks to check gradients
def check_grad_hook(name):
    def hook(grad):
        if torch.isnan(grad).any():
            print(f"  NaN gradient detected in {name}")
        if torch.isinf(grad).any():
            print(f"  Inf gradient detected in {name}")
        print(f"  Gradient {name}: min={grad.min().item():.6f}, max={grad.max().item():.6f}, mean={grad.mean().item():.6f}")
        return grad
    return hook

# Register hooks on policy parameters
for name, param in sac_agent.policy.named_parameters():
    if param.requires_grad:
        param.register_hook(check_grad_hook(f"policy.{name}"))

# Run update
try:
    qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha = sac_agent.update_parameters(
        memory, args.sac_batch_size, updates=0
    )
    print(f"Update completed")
    print(f"  QF1 loss: {qf1_loss}")
    print(f"  QF2 loss: {qf2_loss}")
    print(f"  Policy loss: {policy_loss}")
except Exception as e:
    print(f"Update failed: {e}")
    import traceback
    traceback.print_exc()

"""
Debug Critic training NaN issue
"""
import torch
import torch.nn.functional as F
from env import Box
from model import CirclePF, CirclePB
from sampling import sample_trajectories, evaluate_backward_logprobs, sample_actions
from sac_replay_memory import ReplayMemory, trajectories_to_transitions
from sac import SAC
from env import get_last_states
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

env = Box(dim=2, delta=0.1, epsilon=0.0, device_str=device, reward_type="baseline")

args = argparse.Namespace(
    device=device, dim=2, delta=0.1, hidden_dim=64, n_hidden=2,
    n_components=4, n_components_s0=2, beta_min=1e-2, beta_max=1e2,
    lr=3e-4, tau=0.005, sac_alpha=0.2, automatic_entropy_tuning=False,
    target_update_interval=1, Critic_hidden_size=256,
    gamma_scheduler=0.99, scheduler_milestone=10000,
    replay_size=10000, sac_batch_size=64,
)

policy = CirclePF(
    hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,
    n_components=args.n_components, n_components_s0=args.n_components_s0,
    beta_min=args.beta_min, beta_max=args.beta_max,
).to(device)

bw_model = CirclePB(
    hidden_dim=args.hidden_dim, n_hidden=args.n_hidden, uniform=False,
    n_components=args.n_components, beta_min=args.beta_min, beta_max=args.beta_max,
).to(device)

# Create replay buffer
memory = ReplayMemory(args.replay_size, seed=42, device=device)
for i in range(5):
    trajs, acts, _, _ = sample_trajectories(env, policy, 16)
    last_sts = get_last_states(env, trajs)
    logrews = env.reward(last_sts).log()
    _, all_bw_lps = evaluate_backward_logprobs(env, bw_model, trajs)
    sts, acts_t, rews, next_sts, dns = trajectories_to_transitions(
        trajs, acts, all_bw_lps, last_sts, logrews, env
    )
    memory.push_batch(sts, acts_t, rews, next_sts, dns)

print(f"Memory size: {len(memory)}")

# Create SAC agent
sac_agent = SAC(args, env)

# Manual update with detailed logging
print("\n" + "="*60)
print("Manual SAC update with detailed logging")
print("="*60)

state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(args.sac_batch_size)

print(f"Batch stats:")
print(f"  States: min={state_batch.min().item():.4f}, max={state_batch.max().item():.4f}")
print(f"  Actions: min={action_batch.min().item():.4f}, max={action_batch.max().item():.4f}")
print(f"  Rewards: min={reward_batch.min().item():.4f}, max={reward_batch.max().item():.4f}")
print(f"  Mask: unique values={mask_batch.unique().cpu().numpy()}")

# Check initial critic output
with torch.no_grad():
    qf1_init, qf2_init = sac_agent.critic(state_batch, action_batch)
    print(f"\nInitial critic output:")
    print(f"  QF1: min={qf1_init.min().item():.4f}, max={qf1_init.max().item():.4f}, mean={qf1_init.mean().item():.4f}")
    print(f"  QF1 has NaN: {torch.isnan(qf1_init).any().item()}")

# Compute target Q-value
with torch.no_grad():
    non_sink_mask = ~torch.all(next_state_batch == env.sink_state, dim=-1)
    next_state_action = torch.zeros_like(next_state_batch)
    next_state_log_pi = torch.zeros(next_state_batch.shape[0], device=device)

    if non_sink_mask.any():
        sampled_actions, sampled_log_pi = sample_actions(
            env, sac_agent.policy, next_state_batch[non_sink_mask]
        )
        sampled_actions = torch.where(
            torch.isinf(sampled_actions),
            torch.zeros_like(sampled_actions),
            sampled_actions
        )
        next_state_action[non_sink_mask] = sampled_actions
        next_state_log_pi[non_sink_mask] = sampled_log_pi

    qf1_next_target, qf2_next_target = sac_agent.critic_target(next_state_batch, next_state_action)

    print(f"\nNext state Q-values:")
    print(f"  QF1_next: min={qf1_next_target.min().item():.4f}, max={qf1_next_target.max().item():.4f}")
    print(f"  QF1_next has NaN: {torch.isnan(qf1_next_target).any().item()}")

    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - sac_agent.alpha * next_state_log_pi.unsqueeze(-1)

    print(f"  min_qf_next_target: min={min_qf_next_target.min().item():.4f}, max={min_qf_next_target.max().item():.4f}")
    print(f"  min_qf_next_target has NaN: {torch.isnan(min_qf_next_target).any().item()}")

    next_q_value = reward_batch + mask_batch * sac_agent.gamma * min_qf_next_target

    print(f"\nTarget Q-value:")
    print(f"  next_q_value: min={next_q_value.min().item():.4f}, max={next_q_value.max().item():.4f}")
    print(f"  next_q_value has NaN: {torch.isnan(next_q_value).any().item()}")

# Compute current Q-values
qf1, qf2 = sac_agent.critic(state_batch, action_batch)
print(f"\nCurrent Q-values:")
print(f"  QF1: min={qf1.min().item() if not torch.isnan(qf1).any() else 'nan'}, max={qf1.max().item() if not torch.isnan(qf1).any() else 'nan'}")
print(f"  QF1 has NaN: {torch.isnan(qf1).any().item()}")

if torch.isnan(qf1).any():
    print("\nQF1 is already NaN before computing loss!")
    print("Checking critic parameters:")
    for name, param in sac_agent.critic.named_parameters():
        has_nan = torch.isnan(param).any().item()
        if has_nan:
            print(f"  {name} has NaN!")
else:
    # Compute loss
    qf1_loss = F.mse_loss(qf1, next_q_value)
    qf2_loss = F.mse_loss(qf2, next_q_value)
    print(f"\nLosses:")
    print(f"  QF1 loss: {qf1_loss.item():.4f}")
    print(f"  QF2 loss: {qf2_loss.item():.4f}")

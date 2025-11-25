"""
Debug next state critic target NaN
"""
import torch
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

# Create SAC agent
sac_agent = SAC(args, env)

# Sample batch
state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(args.sac_batch_size)

non_sink_mask = ~torch.all(next_state_batch == env.sink_state, dim=-1)
print(f"Non-sink next states: {non_sink_mask.sum().item()} / {len(non_sink_mask)}")

# Check next_state_batch
print(f"\nNext state batch:")
print(f"  min={next_state_batch.min().item():.4f}, max={next_state_batch.max().item():.4f}")
print(f"  has NaN: {torch.isnan(next_state_batch).any().item()}")
print(f"  has Inf: {torch.isinf(next_state_batch).any().item()}")

# Sample actions for non-sink next states
if non_sink_mask.any():
    print(f"\nSampling actions for {non_sink_mask.sum().item()} non-sink next states")
    sampled_actions, sampled_log_pi = sample_actions(
        env, sac_agent.policy, next_state_batch[non_sink_mask]
    )

    print(f"  Sampled actions: min={sampled_actions.min().item():.4f}, max={sampled_actions.max().item():.4f}")
    print(f"  Sampled actions has NaN: {torch.isnan(sampled_actions).any().item()}")
    print(f"  Sampled actions has Inf: {torch.isinf(sampled_actions).any().item()}")

    # Replace inf
    sampled_actions_clean = torch.where(
        torch.isinf(sampled_actions),
        torch.zeros_like(sampled_actions),
        sampled_actions
    )

    print(f"  After replacing inf: min={sampled_actions_clean.min().item():.4f}, max={sampled_actions_clean.max().item():.4f}")

    # Create full action tensor
    next_state_action = torch.zeros_like(next_state_batch)
    next_state_action[non_sink_mask] = sampled_actions_clean

    print(f"\nFull next_state_action:")
    print(f"  min={next_state_action.min().item():.4f}, max={next_state_action.max().item():.4f}")
    print(f"  has NaN: {torch.isnan(next_state_action).any().item()}")
    print(f"  has Inf: {torch.isinf(next_state_action).any().item()}")

    # Test critic_target with these inputs
    print(f"\nTesting critic_target:")
    with torch.no_grad():
        qf1_next, qf2_next = sac_agent.critic_target(next_state_batch, next_state_action)

        print(f"  QF1_next: min={qf1_next.min().item() if not torch.isnan(qf1_next).any() else 'nan'}, max={qf1_next.max().item() if not torch.isnan(qf1_next).any() else 'nan'}")
        print(f"  QF1_next has NaN: {torch.isnan(qf1_next).any().item()}")

        if torch.isnan(qf1_next).any():
            nan_mask = torch.isnan(qf1_next).squeeze()
            print(f"\n  {nan_mask.sum().item()} outputs are NaN")
            print(f"  NaN indices: {torch.where(nan_mask)[0][:10].cpu().numpy()}")
            print(f"  Corresponding next_states:")
            print(next_state_batch[nan_mask][:5])
            print(f"  Corresponding next_actions:")
            print(next_state_action[nan_mask][:5])

            # Check if these are sink states
            is_sink = torch.all(next_state_batch[nan_mask] == env.sink_state, dim=-1)
            print(f"  Are sink states: {is_sink[:10].cpu().numpy()}")

    # Test with only non-sink states
    print(f"\n\nTesting critic_target with ONLY non-sink next states:")
    non_sink_next_states = next_state_batch[non_sink_mask]
    non_sink_next_actions = next_state_action[non_sink_mask]

    with torch.no_grad():
        qf1_non_sink, qf2_non_sink = sac_agent.critic_target(non_sink_next_states, non_sink_next_actions)

        print(f"  QF1 (non-sink only): min={qf1_non_sink.min().item() if not torch.isnan(qf1_non_sink).any() else 'nan'}, max={qf1_non_sink.max().item() if not torch.isnan(qf1_non_sink).any() else 'nan'}")
        print(f"  QF1 has NaN: {torch.isnan(qf1_non_sink).any().item()}")

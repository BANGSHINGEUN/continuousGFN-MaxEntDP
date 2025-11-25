"""
Debug Critic NaN issue
"""
import torch
from sac_model import QNetwork

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create critic
critic = QNetwork(num_inputs=2, num_actions=2, hidden_dim=256).to(device)

# Test with simple inputs
batch_size = 64
states = torch.randn(batch_size, 2, device=device) * 0.5
actions = torch.randn(batch_size, 2, device=device) * 0.05

print(f"\nStates: min={states.min().item():.4f}, max={states.max().item():.4f}, mean={states.mean().item():.4f}")
print(f"Actions: min={actions.min().item():.4f}, max={actions.max().item():.4f}, mean={actions.mean().item():.4f}")

# Check for NaN in parameters
print("\nChecking critic parameters:")
for name, param in critic.named_parameters():
    has_nan = torch.isnan(param).any().item()
    has_inf = torch.isinf(param).any().item()
    print(f"  {name}: shape={param.shape}, has_nan={has_nan}, has_inf={has_inf}")

# Forward pass
with torch.no_grad():
    q1, q2 = critic(states, actions)

print(f"\nQ1: min={q1.min().item()}, max={q1.max().item()}, mean={q1.mean().item()}")
print(f"Q1 has NaN: {torch.isnan(q1).any().item()}")
print(f"Q1 has Inf: {torch.isinf(q1).any().item()}")

print(f"\nQ2: min={q2.min().item()}, max={q2.max().item()}, mean={q2.mean().item()}")
print(f"Q2 has NaN: {torch.isnan(q2).any().item()}")
print(f"Q2 has Inf: {torch.isinf(q2).any().item()}")

# Now test with actual data from replay buffer
print("\n" + "="*60)
print("Testing with actual replay buffer data")
print("="*60)

from env import Box
from model import CirclePF, CirclePB
from sampling import sample_trajectories, evaluate_backward_logprobs
from sac_replay_memory import ReplayMemory, trajectories_to_transitions
from env import get_last_states

env = Box(dim=2, delta=0.1, epsilon=0.0, device_str=device, reward_type="baseline")

policy = CirclePF(
    hidden_dim=64, n_hidden=2, n_components=4, n_components_s0=2,
    beta_min=1e-2, beta_max=1e2,
).to(device)

bw_model = CirclePB(
    hidden_dim=64, n_hidden=2, uniform=False, n_components=4,
    beta_min=1e-2, beta_max=1e2,
).to(device)

# Sample and create transitions
memory = ReplayMemory(10000, seed=42, device=device)
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

# Sample batch
state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(64)

print(f"\nSampled batch:")
print(f"  States: min={state_batch.min().item():.4f}, max={state_batch.max().item():.4f}")
print(f"  Actions: min={action_batch.min().item():.4f}, max={action_batch.max().item():.4f}")
print(f"  Rewards: min={reward_batch.min().item():.4f}, max={reward_batch.max().item():.4f}")

# Test critic with this data
with torch.no_grad():
    q1_real, q2_real = critic(state_batch, action_batch)

print(f"\nCritic output on real data:")
print(f"  Q1: min={q1_real.min().item() if not torch.isnan(q1_real).any() else 'nan'}, max={q1_real.max().item() if not torch.isnan(q1_real).any() else 'nan'}")
print(f"  Q1 has NaN: {torch.isnan(q1_real).any().item()}")
print(f"  Q1 has Inf: {torch.isinf(q1_real).any().item()}")

# Check which inputs cause NaN
if torch.isnan(q1_real).any():
    nan_mask = torch.isnan(q1_real).squeeze()
    print(f"\n{nan_mask.sum().item()} outputs are NaN")
    print(f"Corresponding states:")
    print(state_batch[nan_mask][:5])
    print(f"Corresponding actions:")
    print(action_batch[nan_mask][:5])

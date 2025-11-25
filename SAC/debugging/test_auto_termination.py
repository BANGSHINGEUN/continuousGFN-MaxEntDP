"""Test automatic termination implementation."""
import torch
from env import Box
from model import CirclePF
from sampling import sample_actions, sample_trajectories

# Create environment
env = Box(
    dim=2,
    delta=0.1,
    epsilon=1e-4,
    device_str="cpu",
    reward_type="corner_squares",
)

# Create model
model = CirclePF(
    hidden_dim=64,
    n_hidden=2,
    n_components=1,
    n_components_s0=1,
    beta_min=0.1,
    beta_max=2.0,
)

print("Testing automatic termination...")
print(f"Environment delta: {env.delta}")
print(f"Environment epsilon: {env.epsilon}")

# Test 1a: Initial state s0
print("\n--- Test 1a: Initial state s0 ---")
states_s0 = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
print("States:")
print(states_s0)
actions, logprobs = sample_actions(env, model, states_s0)
print("Actions:")
print(actions)
terminal_mask = torch.all(actions == -float("inf"), dim=-1)
print(f"Terminal mask: {terminal_mask}")
print(f"Logprobs: {logprobs}")

# Test 1b: States that should NOT terminate
states_safe = torch.tensor([
    [0.5, 0.5],  # middle
    [0.8, 0.5],  # close but not within delta
    [0.7, 0.7],  # safe
])
print("\n--- Test 1b: Safe states (should NOT terminate) ---")
print("States:")
print(states_safe)
actions, logprobs = sample_actions(env, model, states_safe)
print("Actions:")
print(actions)
terminal_mask = torch.all(actions == -float("inf"), dim=-1)
print(f"Terminal mask: {terminal_mask}")
print(f"Logprobs: {logprobs}")

# Test 2: States that SHOULD terminate (within delta of boundary)
states_near_boundary = torch.tensor([
    [0.95, 0.5],   # x close to 1
    [0.5, 0.92],   # y close to 1
    [0.91, 0.91],  # both close to 1
])
print("\n--- Test 2: States near boundary (SHOULD terminate) ---")
print("States:")
print(states_near_boundary)
print(f"Distance to boundary (1 - states):")
print(1 - states_near_boundary)
print(f"Should terminate (any coord >= 1 - delta = {1 - env.delta}):")
print(torch.any(states_near_boundary >= 1 - env.delta, dim=-1))
actions, logprobs = sample_actions(env, model, states_near_boundary)
print("Actions:")
print(actions)
terminal_mask = torch.all(actions == -float("inf"), dim=-1)
print(f"Terminal mask: {terminal_mask}")
print(f"Logprobs: {logprobs}")

# Test 3: Sample full trajectories
print("\n--- Test 3: Sample trajectories ---")
trajectories, actionss, logprobs_total, all_logprobs = sample_trajectories(env, model, 5)
print(f"Trajectories shape: {trajectories.shape}")
print(f"Number of steps: {trajectories.shape[1] - 1}")
print(f"Final states (last non-sink states):")
for i in range(5):
    traj = trajectories[i]
    non_sink = ~torch.all(traj == env.sink_state, dim=-1)
    last_idx = non_sink.nonzero()[-1].item()
    print(f"  Trajectory {i}: {traj[last_idx].numpy()}")

print("\n--- Test 4: Check model output dimensions ---")
test_state = torch.tensor([[0.5, 0.5]])
out = model.to_dist(test_state)
print(f"Model output type: {type(out)}")
print(f"Output is Distribution: {isinstance(out, torch.distributions.Distribution)}")

# Test forward pass
forward_out = model.forward(test_state)
print(f"Forward output length: {len(forward_out)}")
print(f"Forward output shapes: {[o.shape for o in forward_out]}")

print("\n✓ All tests completed!")

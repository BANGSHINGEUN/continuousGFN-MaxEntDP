"""
Debug code to understand trajectories, actionss, and all_bw_logprobs
"""
import torch
import numpy as np
from env import Box, get_last_states
from model import CirclePF, CirclePB
from sampling import sample_trajectories, evaluate_backward_logprobs

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")

# Create simple environment
env = Box(dim=2, delta=0.1, epsilon=0.0, device_str=device, reward_type="baseline")
print(f"Sink state: {env.sink_state}\n")

# Create policy
policy = CirclePF(
    hidden_dim=64,
    n_hidden=2,
    n_components=4,
    n_components_s0=2,
    beta_min=1e-2,
    beta_max=1e2,
).to(device)

# Create backward model
bw_model = CirclePB(
    hidden_dim=64,
    n_hidden=2,
    uniform=False,
    n_components=4,
    beta_min=1e-2,
    beta_max=1e2,
).to(device)

# Sample trajectories
BS = 3  # Small batch for easy visualization
print(f"Sampling {BS} trajectories...\n")
trajectories, actionss, logprobs, all_logprobs = sample_trajectories(env, policy, BS)

print("="*80)
print("SHAPES")
print("="*80)
print(f"trajectories.shape: {trajectories.shape}")
print(f"actionss.shape: {actionss.shape}")
print(f"logprobs.shape: {logprobs.shape}")
print(f"all_logprobs.shape: {all_logprobs.shape}")

# Evaluate backward logprobs
bw_logprobs, all_bw_logprobs = evaluate_backward_logprobs(env, bw_model, trajectories)
print(f"\nbw_logprobs.shape: {bw_logprobs.shape}")
print(f"all_bw_logprobs.shape: {all_bw_logprobs.shape}")

# Get last states and rewards
last_states = get_last_states(env, trajectories)
logrewards = env.reward(last_states).log()
print(f"\nlast_states.shape: {last_states.shape}")
print(f"logrewards.shape: {logrewards.shape}")

print("\n" + "="*80)
print("TRAJECTORY 0 (detailed)")
print("="*80)

traj_0 = trajectories[0]
actions_0 = actionss[0]
all_bw_lp_0 = all_bw_logprobs[0]

print(f"\ntrajectories[0] shape: {traj_0.shape}")
print(f"actionss[0] shape: {actions_0.shape}")
print(f"all_bw_logprobs[0] shape: {all_bw_lp_0.shape}")

print("\nStates in trajectory 0:")
for i, state in enumerate(traj_0):
    is_sink = torch.all(state == env.sink_state).item()
    print(f"  State {i}: {state.cpu().numpy()} {'[SINK]' if is_sink else ''}")

print("\nActions in trajectory 0:")
for i, action in enumerate(actions_0):
    print(f"  Action {i}: {action.cpu().numpy()}")

print("\nBackward log probs for trajectory 0:")
for i, bw_lp in enumerate(all_bw_lp_0):
    print(f"  BW logprob {i}: {bw_lp.item():.4f}")

print(f"\nLast state: {last_states[0].cpu().numpy()}")
print(f"Log reward: {logrewards[0].item():.4f}")

print("\n" + "="*80)
print("ALL TRAJECTORIES")
print("="*80)

for batch_idx in range(BS):
    print(f"\n--- Trajectory {batch_idx} ---")
    traj = trajectories[batch_idx]
    acts = actionss[batch_idx]
    bw_lps = all_bw_logprobs[batch_idx]

    print(f"Num states: {traj.shape[0]}")
    print(f"Num actions: {acts.shape[0]}")
    print(f"Num bw_logprobs: {bw_lps.shape[0]}")

    # Find where sink starts
    sink_idx = None
    for i, state in enumerate(traj):
        if torch.all(state == env.sink_state).item():
            sink_idx = i
            break

    if sink_idx is not None:
        print(f"Sink state appears at index: {sink_idx}")
    else:
        print("No sink state in trajectory")

    print(f"Last state is sink: {torch.all(last_states[batch_idx] == env.sink_state).item()}")

print("\n" + "="*80)
print("DIMENSION ANALYSIS")
print("="*80)

traj_len = trajectories.shape[1]
actions_len = actionss.shape[1]
bw_len = all_bw_logprobs.shape[1]

print(f"\nTrajectory length: {traj_len}")
print(f"Actions length: {actions_len}")
print(f"BW logprobs length: {bw_len}")
print(f"\nDifference:")
print(f"  traj_len - actions_len = {traj_len - actions_len}")
print(f"  traj_len - bw_len = {traj_len - bw_len}")
print(f"  actions_len - bw_len = {actions_len - bw_len}")

print("\n" + "="*80)
print("EXPECTED MAPPING")
print("="*80)

print(f"""
For trajectory with {traj_len} states:
  States indices: 0, 1, 2, ..., {traj_len-1}
  Actions indices: 0, 1, 2, ..., {actions_len-1}
  BW logprobs indices: 0, 1, 2, ..., {bw_len-1}

Transitions should be:
  state[i] --action[i]--> state[i+1]

BW logprobs correspond to:
  (evaluated from evaluate_backward_logprobs)
  Likely corresponds to later states (due to range(traj_len-2, 1, -1))
""")

# Check the range used in evaluate_backward_logprobs
print(f"\nIn evaluate_backward_logprobs, range({traj_len}-2, 1, -1) = range({traj_len-2}, 1, -1)")
print(f"This gives indices: {list(range(traj_len-2, 1, -1))}")
print(f"That's {len(list(range(traj_len-2, 1, -1)))} indices")
print(f"Plus one zero appended = {len(list(range(traj_len-2, 1, -1))) + 1} total")
print(f"Actual all_bw_logprobs length: {bw_len}")

"""
Carefully verify the backward logprob ordering
"""
import torch
import numpy as np

print("="*80)
print("UNDERSTANDING evaluate_backward_logprobs")
print("="*80)

# Simulate the backward logprob computation
# Let's say we have a trajectory: s0 -> s1 -> s2 -> s3 -> s4 -> sink
# trajectory indices:            0     1     2     3     4     5

trajectory_length = 6  # including sink state
print(f"\nTrajectory: s0 -> s1 -> s2 -> s3 -> s4 -> sink")
print(f"Indices:     0     1     2     3     4     5")
print(f"trajectory.shape[1] = {trajectory_length}")

print("\nIn evaluate_backward_logprobs:")
print(f"  range(trajectory.shape[1] - 2, 1, -1) = range({trajectory_length - 2}, 1, -1)")
print(f"  = {list(range(trajectory_length - 2, 1, -1))}")

all_logprobs = []
for i in range(trajectory_length - 2, 1, -1):
    print(f"\n  Step i={i}:")
    print(f"    current_states = trajectories[:, {i}]     = s{i}")
    print(f"    previous_states = trajectories[:, {i-1}]   = s{i-1}")
    print(f"    Computes: P_B(s{i-1} | s{i})")
    all_logprobs.append(f"P_B(s{i-1} | s{i})")

print(f"\n  Then appends zero")
all_logprobs.append("0")

print(f"\n  all_logprobs before flip: {all_logprobs}")
print(f"  all_logprobs after flip:  {list(reversed(all_logprobs))}")

print("\n" + "="*80)
print("INDEXING AFTER FLIP")
print("="*80)

flipped = list(reversed(all_logprobs))
print("\nAfter flip, all_bw_logprobs has length", len(flipped))
for idx, prob in enumerate(flipped):
    print(f"  all_bw_logprobs[{idx}] = {prob}")

print("\n" + "="*80)
print("MAPPING TO TRANSITIONS")
print("="*80)

print("\nIn trajectories_to_transitions:")
print("  bw_length = all_bw_logprobs.shape[1] = 4")
print("  states = trajectories[:, :bw_length, :]        = [:, :4, :]  = [s0, s1, s2, s3]")
print("  next_states = trajectories[:, 1:bw_length+1, :] = [:, 1:5, :] = [s1, s2, s3, s4]")
print("  rewards = all_bw_logprobs                       = [0, P_B(s1|s2), P_B(s2|s3), P_B(s3|s4)]")

print("\n" + "="*80)
print("CHECKING EACH TRANSITION")
print("="*80)

transitions = [
    (0, "s0", "s1", "0"),
    (1, "s1", "s2", "P_B(s1|s2)"),
    (2, "s2", "s3", "P_B(s2|s3)"),
    (3, "s3", "s4", "P_B(s3|s4)"),
]

for t, state, next_state, reward in transitions:
    print(f"\nTransition t={t}:")
    print(f"  state: {state}")
    print(f"  next_state: {next_state}")
    print(f"  reward: {reward}")
    print(f"  Question: Is reward = P_B(state | next_state)?")
    if reward == f"P_B({state}|{next_state})":
        print(f"  Answer: YES ✓ reward = P_B({state} | {next_state})")
    else:
        print(f"  Answer: NO ✗ reward = {reward}, but should be P_B({state} | {next_state})")

print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)
print("✓ For transitions t=1,2,3: reward = P_B(s_t | s_{t+1}) ✓ CORRECT")
print("✗ For transition t=0: reward = 0, but should be P_B(s0 | s1)")
print("\nThe first transition (from s0) has reward=0, which might be intentional")
print("since we don't learn P_B(s0 | s1) in the backward model.")

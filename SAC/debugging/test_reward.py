import torch
import numpy as np
from env import Box

env = Box(dim=2, delta=0.25, reward_type='baseline', device_str='cpu')

# Test reward at different positions
test_points = torch.tensor([
    [0.0, 0.0],   # Start
    [0.5, 0.5],   # Middle
    [0.9, 0.9],   # Near target
    [0.95, 0.95], # Very near target
    [0.99, 0.99], # Almost at target
])

print("Testing baseline reward function:")
print("=" * 60)
for point in test_points:
    reward = env.reward(point.unsqueeze(0))
    log_reward = reward.log()
    print(f"Position {point.numpy()}: reward = {reward.item():.6f}, log = {log_reward.item():.6f}")

print("\nReward function definition:")
print("R0 + (0.25 < |x-0.5|).prod() * R1 + ((0.3 < |x-0.5|) * (|x-0.5| < 0.4)).prod() * R2")
print(f"R0 = {env.R0}, R1 = {env.R1}, R2 = {env.R2}")

"""
Check learning rate scheduler behavior
"""
import torch
from torch.optim import Adam

# Simulate the scheduler behavior
model = torch.nn.Linear(2, 2)
optimizer = Adam(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[i * 2500 for i in range(1, 10)],
    gamma=0.5,
)

print("Milestones:", [i * 2500 for i in range(1, 10)])
print("Gamma:", 0.5)
print("\nLearning rate over update steps:\n")

# Track LR over many steps
updates_per_iteration = 1
iterations = 6000

for iteration in range(iterations):
    for _ in range(updates_per_iteration):
        scheduler.step()

    if iteration % 500 == 0 or iteration in [2499, 2500, 2501, 5000, 7500]:
        current_lr = optimizer.param_groups[0]['lr']
        total_steps = (iteration + 1) * updates_per_iteration
        print(f"Iteration {iteration:5d} (step {total_steps:5d}): LR = {current_lr:.8f}")

import torch
import numpy as np
import random


class TrajectoryReplayMemory:
    """
    Replay buffer for trajectories with variable lengths.
    
    Stores:
    - trajectories: shape (capacity, max_traj_len, 2)
    - samples: shape (capacity, max_traj_len - 1, 2)
    
    Uses lazy initialization like SAC replay memory - buffers are allocated on first push.
    Automatically handles varying max_traj_len across batches by tracking and updating.
    """
    
    def __init__(self, capacity, seed, device='cpu'):
        """
        Args:
            capacity: Maximum number of trajectories to store
            seed: Random seed for reproducibility
            device: Device to store tensors on
        """
        random.seed(seed)
        np.random.seed(seed)
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        
        # Will be initialized on first push
        self.trajectories = None
        self.samples = None
        self.max_traj_len = 0
        
    def push_batch(self, trajectories, samples):
        """
        Push a batch of trajectories to the replay buffer.
        
        Args:
            trajectories: tensor of shape (batch_size, traj_len, 2)
            samples: tensor of shape (batch_size, traj_len - 1, 2)
        """
        batch_size = trajectories.shape[0]
        traj_len = trajectories.shape[1]
        
        # Move to device if needed
        trajectories = trajectories.to(self.device)
        samples = samples.to(self.device)
        
        # Initialize or resize buffers if needed
        if self.trajectories is None:
            # First push - initialize buffers
            self.max_traj_len = traj_len
            self.trajectories = torch.full(
                (self.capacity, traj_len, 2), 
                -float('inf'), 
                dtype=trajectories.dtype, 
                device=self.device
            )
            self.samples = torch.full(
                (self.capacity, traj_len - 1, 2), 
                -float('inf'), 
                dtype=samples.dtype, 
                device=self.device
            )
        elif traj_len > self.max_traj_len:
            # Need to resize buffers to accommodate longer trajectories
            old_max_len = self.max_traj_len
            self.max_traj_len = traj_len
            
            # Create new larger buffers
            new_trajectories = torch.full(
                (self.capacity, traj_len, 2), 
                -float('inf'), 
                dtype=self.trajectories.dtype, 
                device=self.device
            )
            new_samples = torch.full(
                (self.capacity, traj_len - 1, 2), 
                -float('inf'), 
                dtype=self.samples.dtype, 
                device=self.device
            )
            
            # Copy old data to new buffers
            new_trajectories[:, :old_max_len, :] = self.trajectories
            new_samples[:, :old_max_len - 1, :] = self.samples
            
            self.trajectories = new_trajectories
            self.samples = new_samples
        
        # Calculate indices
        end_pos = self.position + batch_size
        
        if end_pos <= self.capacity:
            # No wrap around
            # Reset to -inf first (to handle varying lengths)
            self.trajectories[self.position:end_pos] = -float('inf')
            self.samples[self.position:end_pos] = -float('inf')
            
            # Write actual data
            self.trajectories[self.position:end_pos, :traj_len, :] = trajectories
            self.samples[self.position:end_pos, :traj_len - 1, :] = samples
        else:
            # Wrap around
            first_part = self.capacity - self.position
            
            # First part
            self.trajectories[self.position:] = -float('inf')
            self.samples[self.position:] = -float('inf')
            self.trajectories[self.position:, :traj_len, :] = trajectories[:first_part]
            self.samples[self.position:, :traj_len - 1, :] = samples[:first_part]
            
            # Second part
            second_part = batch_size - first_part
            self.trajectories[:second_part] = -float('inf')
            self.samples[:second_part] = -float('inf')
            self.trajectories[:second_part, :traj_len, :] = trajectories[first_part:]
            self.samples[:second_part, :traj_len - 1, :] = samples[first_part:]
        
        self.position = end_pos % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
    
    def sample(self, batch_size):
        """
        Sample a random batch from the buffer.
        
        Returns:
            trajectories: tensor of shape (batch_size, max_traj_len, 2)
            samples: tensor of shape (batch_size, max_traj_len - 1, 2)
        """
        indices = np.random.choice(self.size, batch_size, replace=False)
        indices = torch.from_numpy(indices).to(self.device)
        
        return (
            self.trajectories[indices],
            self.samples[indices],
        )
    
    def __len__(self):
        return self.size


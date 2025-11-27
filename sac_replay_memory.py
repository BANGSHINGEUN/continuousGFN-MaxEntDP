import random
import numpy as np
import torch
import os
import pickle


def trajectories_to_transitions(trajectories, actionss, all_bw_logprobs, logrewards, env):
    """
    Convert trajectories to transitions for replay buffer.

    Args:
        trajectories: tensor of shape (batch_size, trajectory_length, dim)
        actionss: tensor of shape (batch_size, trajectory_length, dim)
        all_bw_logprobs: tensor of shape (batch_size, trajectory_length)
        last_states: tensor of shape (batch_size, dim)
        logrewards: tensor of shape (batch_size,)
        env: environment object

    Returns:
        Tuple of (states, actions, rewards, next_states, dones) as tensors
    """

    # Extract states and next_states for intermediate transitions
    # Match the length to all_bw_logprobs

    states = trajectories[:, :-1, :]  
    next_states = trajectories[:, 1:, :] 
    is_not_sink = torch.all(states != env.sink_state, dim=-1)
    is_next_sink = torch.all(next_states == env.sink_state, dim=-1)
    last_state = is_not_sink & is_next_sink
    dones = torch.zeros_like(last_state, dtype=torch.float32)  # (batch_size, bw_length)
    dones[last_state] = 1.0
    dones = dones[:, 1:]
    rewards = all_bw_logprobs
    rewards = torch.where(last_state[:,1:], rewards + logrewards.unsqueeze(1), rewards)
    states = states[:, :-1, :]  
    next_states = next_states[:, :-1, :] 
    actions = actionss[:, :-1, :] 
    # Check which rewards are valid (not inf/nan)
    is_valid = torch.isfinite(rewards)  # (batch_size, bw_length)

    # Flatten batch and time dimensions for transitions
    states_flat = states[is_valid]
    actions_flat = actions[is_valid]
    rewards_flat = rewards[is_valid]
    next_states_flat = next_states[is_valid]
    dones_flat = dones[is_valid]

    return states_flat, actions_flat, rewards_flat, next_states_flat, dones_flat


class ReplayMemory:
    def __init__(self, capacity, seed, device='cpu'):
        random.seed(seed)
        np.random.seed(seed)
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0

        # Will be initialized on first push
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None

    def push_batch(self, states, actions, rewards, next_states, dones):
        """
        Push a batch of transitions to the replay buffer.

        Args:
            states: tensor of shape (batch_size, state_dim)
            actions: tensor of shape (batch_size, action_dim)
            rewards: tensor of shape (batch_size,)
            next_states: tensor of shape (batch_size, state_dim)
            dones: tensor of shape (batch_size,)
        """
        batch_size = states.shape[0]

        # Initialize buffers on first push
        if self.states is None:
            state_dim = states.shape[1]
            action_dim = actions.shape[1]
            self.states = torch.zeros((self.capacity, state_dim), dtype=states.dtype, device=self.device)
            self.actions = torch.zeros((self.capacity, action_dim), dtype=actions.dtype, device=self.device)
            self.rewards = torch.zeros(self.capacity, dtype=rewards.dtype, device=self.device)
            self.next_states = torch.zeros((self.capacity, state_dim), dtype=next_states.dtype, device=self.device)
            self.dones = torch.zeros(self.capacity, dtype=dones.dtype, device=self.device)

        # Move to device if needed
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Calculate indices
        end_pos = self.position + batch_size

        if end_pos <= self.capacity:
            # No wrap around
            self.states[self.position:end_pos] = states
            self.actions[self.position:end_pos] = actions
            self.rewards[self.position:end_pos] = rewards
            self.next_states[self.position:end_pos] = next_states
            self.dones[self.position:end_pos] = dones
        else:
            # Wrap around
            first_part = self.capacity - self.position
            self.states[self.position:] = states[:first_part]
            self.actions[self.position:] = actions[:first_part]
            self.rewards[self.position:] = rewards[:first_part]
            self.next_states[self.position:] = next_states[:first_part]
            self.dones[self.position:] = dones[:first_part]

            second_part = batch_size - first_part
            self.states[:second_part] = states[first_part:]
            self.actions[:second_part] = actions[first_part:]
            self.rewards[:second_part] = rewards[first_part:]
            self.next_states[:second_part] = next_states[first_part:]
            self.dones[:second_part] = dones[first_part:]

        self.position = end_pos % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        """Sample a random batch from the buffer and return as numpy arrays for compatibility."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        indices = torch.from_numpy(indices).to(self.device)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices].unsqueeze(-1),  # (batch_size,) -> (batch_size, 1)
            self.next_states[indices],
            self.dones[indices].unsqueeze(-1),  # (batch_size,) -> (batch_size, 1)
        )

    def __len__(self):
        return self.size

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        buffer_dict = {
            'states': self.states[:self.size].cpu() if self.states is not None else None,
            'actions': self.actions[:self.size].cpu() if self.actions is not None else None,
            'rewards': self.rewards[:self.size].cpu() if self.rewards is not None else None,
            'next_states': self.next_states[:self.size].cpu() if self.next_states is not None else None,
            'dones': self.dones[:self.size].cpu() if self.dones is not None else None,
            'position': self.position,
            'size': self.size,
        }

        with open(save_path, 'wb') as f:
            pickle.dump(buffer_dict, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            buffer_dict = pickle.load(f)

        self.position = buffer_dict['position']
        self.size = buffer_dict['size']

        if buffer_dict['states'] is not None:
            state_dim = buffer_dict['states'].shape[1]
            action_dim = buffer_dict['actions'].shape[1]

            self.states = torch.zeros((self.capacity, state_dim), device=self.device)
            self.actions = torch.zeros((self.capacity, action_dim), device=self.device)
            self.rewards = torch.zeros(self.capacity, device=self.device)
            self.next_states = torch.zeros((self.capacity, state_dim), device=self.device)
            self.dones = torch.zeros(self.capacity, device=self.device)

            self.states[:self.size] = buffer_dict['states'].to(self.device)
            self.actions[:self.size] = buffer_dict['actions'].to(self.device)
            self.rewards[:self.size] = buffer_dict['rewards'].to(self.device)
            self.next_states[:self.size] = buffer_dict['next_states'].to(self.device)
            self.dones[:self.size] = buffer_dict['dones'].to(self.device)

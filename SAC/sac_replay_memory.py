import random
import numpy as np
import torch
import os
import pickle


def trajectories_to_transitions(trajectories, actionss, all_bw_logprobs, last_states, logrewards, env):
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
    batch_size = trajectories.shape[0]
    traj_length = trajectories.shape[1]
    device = trajectories.device

    # all_bw_logprobs length can be shorter than trajectories due to how evaluate_backward_logprobs works
    # It computes backward probs for range(traj_length-2, 1, -1) and appends one zero
    bw_length = all_bw_logprobs.shape[1]

    # Extract states and next_states for intermediate transitions
    # Match the length to all_bw_logprobs
    states = trajectories[:, :bw_length, :]  # (batch_size, bw_length, dim)
    next_states = trajectories[:, 1:bw_length+1, :]  # (batch_size, bw_length, dim)
    actions = actionss[:, :bw_length, :]  # (batch_size, bw_length, dim)
    rewards = all_bw_logprobs  # (batch_size, bw_length)

    # Check which states are sink states
    is_sink = torch.all(states == env.sink_state, dim=-1)  # (batch_size, bw_length)
    is_next_sink = torch.all(next_states == env.sink_state, dim=-1)  # (batch_size, bw_length)

    # Check which rewards are valid (not inf/nan)
    is_valid_reward = torch.isfinite(rewards)  # (batch_size, bw_length)

    # Create mask: valid transitions are where current state is not sink AND reward is finite
    valid_mask = ~is_sink & is_valid_reward  # (batch_size, bw_length)

    # Done flag: 1 if next_state is sink, 0 otherwise (but not done for intermediate transitions)
    dones = torch.zeros_like(is_next_sink, dtype=torch.float32)  # (batch_size, bw_length)

    # Flatten batch and time dimensions for intermediate transitions
    states_flat = states[valid_mask]  # (num_valid, dim)
    next_states_flat = next_states[valid_mask]  # (num_valid, dim)
    actions_flat = actions[valid_mask]  # (num_valid, dim)
    rewards_flat = rewards[valid_mask]  # (num_valid,)
    dones_flat = dones[valid_mask]  # (num_valid,)

    # Add terminal transitions: last_states -> sink_state with logrewards as reward
    # Find the last non-sink state for each trajectory
    last_non_sink_mask = ~torch.all(last_states == env.sink_state, dim=-1)  # (batch_size,)

    terminal_states = last_states[last_non_sink_mask]  # (num_terminal, dim)
    terminal_next_states = env.sink_state.unsqueeze(0).expand(terminal_states.shape[0], -1)  # (num_terminal, dim)
    # For terminal action, use the last action in the trajectory (or zero if it's -inf)
    terminal_actions = actionss[:, -1, :][last_non_sink_mask]  # (num_terminal, dim)
    terminal_rewards = logrewards[last_non_sink_mask]  # (num_terminal,)
    terminal_dones = torch.ones(terminal_states.shape[0], dtype=torch.float32, device=device)  # (num_terminal,)

    # Concatenate intermediate and terminal transitions
    all_states = torch.cat([states_flat, terminal_states], dim=0)
    all_next_states = torch.cat([next_states_flat, terminal_next_states], dim=0)
    all_actions = torch.cat([actions_flat, terminal_actions], dim=0)
    all_rewards = torch.cat([rewards_flat, terminal_rewards], dim=0)
    all_dones = torch.cat([dones_flat, terminal_dones], dim=0)

    return all_states, all_actions, all_rewards, all_next_states, all_dones


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

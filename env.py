import torch
from torch.distributions import MultivariateNormal
from rewards import get_reward_function, get_Z_function


class Box:
    """D-dimensional box with lower bound 0 and upper bound 1. A maximum step size 0<delta<1 defines
    the maximum unidimensional step size in each dimension.
    """

    def __init__(
        self,
        dim=2,
        delta=0.1,
        R0=0.1,
        R1=1.0,
        R2=2.0,
        reward_type="baseline",
        reward_debug=False,
        device_str="cpu",
        verify_actions=False,
    ):
        # Set verify_actions to False to disable action verification for faster step execution.
        self.dim = dim
        self.delta = delta
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.terminal_action = torch.full((dim,), -float("inf"), device=self.device)
        self.sink_state = torch.full((dim,), -float("inf"), device=self.device)
        self.verify_actions = verify_actions

        # Reward parameters
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.reward_type = reward_type
        self.reward_debug = reward_debug

        # Basic parameters passed to reward functions
        # Reward-specific parameters (radius, sigma, etc.) are defined in rewards.py
        self.reward_params = {
            'R0': R0,
            'R1': R1,
            'R2': R2,
            'delta': delta,
        }

        # Get reward and Z functions
        if self.reward_debug:
            self.reward_fn = get_reward_function('debug')
            self.Z_fn = get_Z_function('debug')
        else:
            self.reward_fn = get_reward_function(reward_type)
            self.Z_fn = get_Z_function(reward_type)

    def is_terminal_action_mask(self, actions):
        """Return a mask of terminal actions."""
        return torch.all(actions == self.terminal_action, dim=-1)

    def step(self, states, actions) :
        """Take a step in the environment. The states can include the sink state [-inf, ..., -inf].
        In which case, the corresponding actions are ignored."""
        # First, select the states that are not the sink state.
        non_sink_mask = ~torch.all(states == self.sink_state, dim=-1)
        non_sink_states = states[non_sink_mask]
        non_sink_actions = actions[non_sink_mask]
        # Then, select states and actions not corresponding to terminal actions, for the non sink states and actions.
        non_terminal_mask = ~self.is_terminal_action_mask(non_sink_actions)
        non_terminal_states = non_sink_states[non_terminal_mask]
        non_terminal_actions = non_sink_actions[non_terminal_mask]
        # Then, take a step and store that in a new tensor.
        new_states = torch.full_like(states, -float("inf"))
        non_sink_new_states = new_states[non_sink_mask]
        non_sink_new_states[non_terminal_mask] = (
            non_terminal_states + non_terminal_actions
        )
        new_states[non_sink_mask] = non_sink_new_states
        # Finally, return the new states.
        return new_states

    def reward(self, final_states):
        """Compute reward for final states using the configured reward function."""
        return self.reward_fn(final_states, **self.reward_params)

    @property
    def Z(self):
        """Compute partition function using the configured Z function."""
        return self.Z_fn(self.dim, **self.reward_params)


def get_last_states(env: Box, trajectories):
    """Get last states from trajectories.
    Args:
        trajectories: A tensor of trajectories
    Returns:
        last_states: A tensor of last states
    """
    non_sink = ~torch.all(trajectories == env.sink_state, dim=-1)

    mask = torch.zeros_like(non_sink).bool()
    mask.scatter_(1, non_sink.cumsum(dim=-1).argmax(dim=-1, keepdim=True), True)

    return trajectories[mask]
        
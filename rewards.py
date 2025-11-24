"""
Reward functions for the Box environment.

Each reward function takes final_states tensor of shape (batch_size, dim)
and returns rewards of shape (batch_size,).
"""

import torch
import math


def reward_baseline(final_states, R0, R1, R2, **kwargs):
    """
    Original corner-band reward structure.

    High reward in corners and specific bands away from center.
    """
    ax = (final_states - 0.5).abs()
    reward = (
        R0 + (0.25 < ax).prod(-1) * R1 + ((0.3 < ax) * (ax < 0.4)).prod(-1) * R2
    )
    return reward


def Z_baseline(dim, R0, R1, R2, **kwargs):
    """Analytical partition function for baseline reward."""
    return (
        R0
        + (2 * 0.25) ** dim * R1
        + (2 * 0.1) ** dim * R2
    )


def reward_ring(final_states, R0, R2, ring_radius=0.3, ring_sigma=0.03, **kwargs):
    """
    Ring-shaped reward: high reward at a specific radius from center.

    Args:
        ring_radius: radius of the ring (default 0.3)
        ring_sigma: thickness of the ring (default 0.03)
    """
    y = final_states - 0.5
    r = y.norm(dim=-1)

    radial = torch.exp(-0.5 * ((r - ring_radius) / ring_sigma) ** 2)

    reward = R0 + R2 * radial
    return reward


def Z_ring(dim, R0, R2, ring_radius=0.3, ring_sigma=0.03, **kwargs):
    """
    Numerical approximation of partition function for ring reward.

    For d dimensions, the volume element at radius r is proportional to r^(d-1).
    """
    # For small sigma, the integral is approximately the Gaussian normalization
    # times the surface area of the sphere at ring_radius
    if dim == 2:
        # 2D: circumference = 2π * r
        surface_area = 2 * math.pi * ring_radius
    elif dim == 3:
        # 3D: surface area = 4π * r^2
        surface_area = 4 * math.pi * ring_radius ** 2
    else:
        # General d-dimensional sphere surface area
        # S_{d-1}(r) = 2π^(d/2) / Γ(d/2) * r^(d-1)
        from scipy.special import gamma
        surface_area = (2 * math.pi ** (dim / 2) / gamma(dim / 2)) * ring_radius ** (dim - 1)

    # Approximate integral: Gaussian width * surface area * peak height
    # Since the Gaussian integral in 1D is sqrt(2π) * sigma, we use this as the width
    ring_contribution = R2 * math.sqrt(2 * math.pi) * ring_sigma * surface_area

    # Background contribution (volume of unit hypercube is 1)
    background = R0

    return background + ring_contribution


def reward_angular_ring(final_states, R0, R2, ring_radius=0.3, ring_sigma=0.03,
                       num_lobes=6, **kwargs):
    """
    Ring with angular modulation: multiple peaks around the ring.

    Args:
        ring_radius: radius of the ring
        ring_sigma: thickness of the ring
        num_lobes: number of angular peaks (default 6)
    """
    dim = final_states.shape[-1]
    if dim < 2:
        raise ValueError("angular_ring requires dim >= 2")

    y = final_states - 0.5
    r = y.norm(dim=-1)

    # Angular component (2D projection)
    theta = torch.atan2(y[..., 1], y[..., 0])  # (-pi, pi]

    # Radial component
    radial = torch.exp(-0.5 * ((r - ring_radius) / ring_sigma) ** 2)

    # Angular component: creates num_lobes peaks
    angular = 0.5 * (1.0 + torch.cos(num_lobes * theta))  # [0, 1]

    reward = R0 + R2 * radial * angular
    return reward


def Z_angular_ring(dim, R0, R2, ring_radius=0.3, ring_sigma=0.03, num_lobes=6, **kwargs):
    """
    Partition function for angular ring reward.

    The angular modulation averages to 0.5 over a full circle.
    """
    base_Z = Z_ring(dim, R0, R2, ring_radius, ring_sigma)
    # Angular component averages to 0.5, so scale the ring contribution by 0.5
    return R0 + (base_Z - R0) * 0.5


def reward_multi_ring(final_states, R0, R1, radii=None, sigmas=None, **kwargs):
    """
    Multiple concentric rings with different radii.

    Args:
        radii: list of ring radii (default [0.2, 0.4, 0.6])
        sigmas: list of ring thicknesses (default [0.02, 0.03, 0.015])
    """
    device = final_states.device

    if radii is None:
        radii = [0.2, 0.4, 0.6]
    if sigmas is None:
        sigmas = [0.02, 0.03, 0.015]

    radii_t = torch.tensor(radii, device=device)
    sigmas_t = torch.tensor(sigmas, device=device)

    y = final_states - 0.5
    r = y.norm(dim=-1)  # (batch,)

    # (batch, 1) - (1, num_rings) -> (batch, num_rings)
    diff = r.unsqueeze(-1) - radii_t.unsqueeze(0)
    radial_terms = torch.exp(-0.5 * (diff / sigmas_t.unsqueeze(0)) ** 2)

    rings_sum = radial_terms.sum(dim=-1)

    reward = R0 + R1 * rings_sum
    return reward


def Z_multi_ring(dim, R0, R1, radii=None, sigmas=None, **kwargs):
    """Partition function for multi-ring reward."""
    if radii is None:
        radii = [0.2, 0.4, 0.6]
    if sigmas is None:
        sigmas = [0.02, 0.03, 0.015]

    total_ring_contribution = 0.0
    for radius, sigma in zip(radii, sigmas):
        # Each ring contributes independently
        ring_Z = Z_ring(dim, 0.0, 1.0, radius, sigma)
        total_ring_contribution += ring_Z

    return R0 + R1 * total_ring_contribution


def reward_curve(final_states, R0, R2, curve_sigma=0.02, noise_sigma=0.1, **kwargs):
    """
    1D curve manifold: reward concentrated near a sinusoidal curve.

    For dim >= 2: curve is in (x1, x2) plane: x2 ≈ 0.5 + 0.25*sin(4π(x1-0.5))
    Other dimensions should be near 0.5.

    Args:
        curve_sigma: thickness of the curve (default 0.02)
        noise_sigma: tolerance for extra dimensions (default 0.1)
    """
    dim = final_states.shape[-1]
    if dim < 2:
        raise ValueError("curve reward requires dim >= 2")

    x = final_states
    x1 = x[..., 0]
    x2 = x[..., 1]

    # Target curve in [0,1] space
    target_x2 = 0.5 + 0.25 * torch.sin(4 * torch.pi * (x1 - 0.5))

    # Distance from curve
    dist2 = ((x2 - target_x2) ** 2) / (2 * curve_sigma ** 2)

    # Penalty for extra dimensions being away from 0.5
    if dim > 2:
        z = x[..., 2:]
        dist2 = dist2 + ((z - 0.5) ** 2).sum(dim=-1) / (2 * noise_sigma ** 2)

    reward = R0 + R2 * torch.exp(-dist2)
    return reward


def Z_curve(dim, R0, R2, curve_sigma=0.02, noise_sigma=0.1, **kwargs):
    """
    Partition function for curve reward.

    Approximation: length of curve × thickness × extra dimension volume.
    """
    # Curve length: approximately 1 (parametrized by x1 from 0 to 1)
    curve_length = 1.0

    # Thickness in x2 direction
    thickness = math.sqrt(2 * math.pi) * curve_sigma

    # Volume in extra dimensions (Gaussian around 0.5)
    if dim > 2:
        extra_volume = (math.sqrt(2 * math.pi) * noise_sigma) ** (dim - 2)
    else:
        extra_volume = 1.0

    curve_contribution = R2 * curve_length * thickness * extra_volume

    return R0 + curve_contribution


def reward_gaussian_mixture(final_states, R0, R1, mixture_sigma=0.05,
                           means_2d=None, **kwargs):
    """
    Gaussian mixture: multiple Gaussian peaks at specified locations.

    Args:
        mixture_sigma: width of each Gaussian component
        means_2d: list of 2D positions for modes (default: 4 corners + center)
    """
    dim = final_states.shape[-1]
    device = final_states.device

    # Default: 4 corners + center
    if means_2d is None:
        means_2d = [
            [0.2, 0.2],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.8, 0.8],
            [0.5, 0.5],
        ]

    means_2d_t = torch.tensor(means_2d, device=device)  # (K, 2)
    K = means_2d_t.shape[0]

    # Extend means to full dimensionality
    if dim > 2:
        # Extra dimensions centered at 0.5
        pad = torch.full((K, dim - 2), 0.5, device=device)
        means = torch.cat([means_2d_t, pad], dim=1)  # (K, dim)
    elif dim == 2:
        means = means_2d_t
    elif dim == 1:
        # For 1D, use simple modes
        means = torch.tensor([[0.2], [0.5], [0.8]], device=device)
        K = means.shape[0]
    else:
        raise ValueError(f"Unsupported dim: {dim}")

    x_exp = final_states.unsqueeze(1)  # (B, 1, dim)
    means_exp = means.unsqueeze(0)  # (1, K, dim)

    diff = x_exp - means_exp  # (B, K, dim)
    dist2 = (diff ** 2).sum(dim=-1)  # (B, K)

    comps = torch.exp(-0.5 * dist2 / (mixture_sigma ** 2))  # (B, K)
    mixture = comps.sum(dim=-1)  # (B,)

    reward = R0 + R1 * mixture
    return reward


def Z_gaussian_mixture(dim, R0, R1, mixture_sigma=0.05, means_2d=None, **kwargs):
    """
    Partition function for Gaussian mixture reward.

    Each component integrates to (sqrt(2π) * sigma)^dim.
    """
    if means_2d is None:
        K = 5  # default number of components
    else:
        K = len(means_2d)

    if dim == 1:
        K = 3  # special case for 1D

    # Each Gaussian component integrates to (sqrt(2π) * sigma)^dim
    single_gaussian_integral = (math.sqrt(2 * math.pi) * mixture_sigma) ** dim

    total_mixture_contribution = R1 * K * single_gaussian_integral

    return R0 + total_mixture_contribution


def reward_debug(final_states, delta, **kwargs):
    """
    Debug reward: uniform inside a ball of radius delta, zero outside.

    Used for testing and validation.
    """
    device = final_states.device
    reward = torch.ones(final_states.shape[0], device=device)
    reward[final_states.norm(dim=-1) > delta] = 1e-8
    return reward


def Z_debug(dim, delta, **kwargs):
    """Partition function for debug reward."""
    if dim != 2:
        raise NotImplementedError("Debug Z only implemented for dim=2")
    return math.pi * delta ** 2 / 4.0


# Registry mapping reward types to their functions
REWARD_FUNCTIONS = {
    'baseline': reward_baseline,
    'ring': reward_ring,
    'angular_ring': reward_angular_ring,
    'multi_ring': reward_multi_ring,
    'curve': reward_curve,
    'gaussian_mixture': reward_gaussian_mixture,
    'debug': reward_debug,
}

Z_FUNCTIONS = {
    'baseline': Z_baseline,
    'ring': Z_ring,
    'angular_ring': Z_angular_ring,
    'multi_ring': Z_multi_ring,
    'curve': Z_curve,
    'gaussian_mixture': Z_gaussian_mixture,
    'debug': Z_debug,
}


def get_reward_function(reward_type):
    """Get reward function by name."""
    if reward_type not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward_type: {reward_type}. "
                        f"Available: {list(REWARD_FUNCTIONS.keys())}")
    return REWARD_FUNCTIONS[reward_type]


def get_Z_function(reward_type):
    """Get partition function by name."""
    if reward_type not in Z_FUNCTIONS:
        raise ValueError(f"Unknown reward_type: {reward_type}. "
                        f"Available: {list(Z_FUNCTIONS.keys())}")
    return Z_FUNCTIONS[reward_type]

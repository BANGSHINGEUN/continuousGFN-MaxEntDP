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


def reward_corner_squares(final_states, R0, R1, R2, **kwargs):
    """
    Corner squares reward: high reward in corners with nested structure.

    Structure (for 2D):
    - 3 corners (excluding bottom-left) each have an outer square of size 0.25 × 0.25
    - Inside each outer square (centered), there's an inner square of size 0.125 × 0.125
    - Outer square: reward R1 (e.g., 10)
    - Inner square (centered in outer square): reward R2 (e.g., 100)
    - Rest: reward 1e-9 (baseline)

    Corner positions:
    - Bottom-right: (1, 0)
    - Top-left: (0, 1)
    - Top-right: (1, 1)
    """
    device = final_states.device
    batch_size = final_states.shape[0]
    dim = final_states.shape[-1]

    if dim != 2:
        raise NotImplementedError("corner_squares only implemented for dim=2")

    # Fixed sizes for corner reward regions
    outer_size = 0.25
    inner_size = 0.125  # outer_size / 2

    # Initialize with baseline reward (1e-9 instead of 0)
    reward = torch.full((batch_size,), 1e-9, device=device)

    x = final_states[:, 0]  # x coordinate
    y = final_states[:, 1]  # y coordinate

    # Define corners and check each one (excluding bottom-left where s0 is)
    corners = [
        (1.0, 0.0),  # bottom-right
        (0.0, 1.0),  # top-left
        (1.0, 1.0),  # top-right
    ]

    for cx, cy in corners:
        # Distance from corner
        if cx == 0.0:
            dx = x
        else:  # cx == 1.0
            dx = 1.0 - x

        if cy == 0.0:
            dy = y
        else:  # cy == 1.0
            dy = 1.0 - y

        # Check if in outer square (size 0.25 × 0.25 from corner)
        in_outer = (dx <= outer_size) & (dy <= outer_size)

        # For inner square, check if centered within outer square
        # Distance from center of outer square
        dx_from_center = torch.abs(dx - outer_size / 2)  # distance from center of outer square
        dy_from_center = torch.abs(dy - outer_size / 2)

        # Inner square is centered in the outer square
        in_inner = (dx_from_center <= inner_size / 2) & (dy_from_center <= inner_size / 2) & in_outer

        # Assign rewards: inner overrides outer
        reward[in_outer] = R1
        reward[in_inner] = R2

    return reward


def Z_corner_squares(dim, R0, R1, R2, **kwargs):
    """Partition function for corner_squares reward with fixed corner size of 0.25."""
    if dim != 2:
        raise NotImplementedError("corner_squares Z only implemented for dim=2")

    # Fixed corner size
    corner_size = 0.25
    inner_size = 0.125

    # Area of each outer square
    outer_square_area = corner_size * corner_size

    # Area of each inner square
    inner_square_area = inner_size * inner_size

    # Area with R1 reward (outer square minus inner square) × 3 corners (excluding bottom-left)
    area_R1 = 3 * (outer_square_area - inner_square_area)

    # Area with R2 reward (inner square) × 3 corners
    area_R2 = 3 * inner_square_area

    # Rest of the space (unit square minus 3 corner squares)
    area_R0 = 1.0 - 3 * outer_square_area

    # Use 1e-9 as baseline instead of R0
    Z = 1e-9 * area_R0 + R1 * area_R1 + R2 * area_R2

    return Z


def reward_two_corners(final_states, R0, R1, R2, **kwargs):
    """
    Two corners reward: high reward in 2 corners with nested structure.

    Structure (for 2D):
    - 2 corners (bottom-right and top-left) each have an outer square of size 0.25 × 0.25
    - Inside each outer square (centered), there's an inner square of size 0.125 × 0.125
    - Outer square: reward R1 (e.g., 0.5)
    - Inner square (centered in outer square): reward R2 (e.g., 2.0)
    - Rest: reward 1e-9 (baseline)

    Corner positions:
    - Bottom-right: (1, 0)
    - Top-left: (0, 1)
    """
    device = final_states.device
    batch_size = final_states.shape[0]
    dim = final_states.shape[-1]

    if dim != 2:
        raise NotImplementedError("two_corners only implemented for dim=2")

    # Fixed sizes for corner reward regions
    outer_size = 0.25
    inner_size = 0.125  # outer_size / 2

    # Initialize with baseline reward (1e-9 instead of 0)
    reward = torch.full((batch_size,), 1e-9, device=device)

    x = final_states[:, 0]  # x coordinate
    y = final_states[:, 1]  # y coordinate

    # Define corners (only 2: bottom-right and top-left)
    corners = [
        (1.0, 0.0),  # bottom-right
        (0.0, 1.0),  # top-left
    ]

    for cx, cy in corners:
        # Distance from corner
        if cx == 0.0:
            dx = x
        else:  # cx == 1.0
            dx = 1.0 - x

        if cy == 0.0:
            dy = y
        else:  # cy == 1.0
            dy = 1.0 - y

        # Check if in outer square (size 0.25 × 0.25 from corner)
        in_outer = (dx <= outer_size) & (dy <= outer_size)

        # For inner square, check if centered within outer square
        # Distance from center of outer square
        dx_from_center = torch.abs(dx - outer_size / 2)  # distance from center of outer square
        dy_from_center = torch.abs(dy - outer_size / 2)

        # Inner square is centered in the outer square
        in_inner = (dx_from_center <= inner_size / 2) & (dy_from_center <= inner_size / 2) & in_outer

        # Assign rewards: inner overrides outer
        reward[in_outer] = R1
        reward[in_inner] = R2

    return reward


def Z_two_corners(dim, R0, R1, R2, **kwargs):
    """Partition function for two_corners reward with fixed corner size of 0.25."""
    if dim != 2:
        raise NotImplementedError("two_corners Z only implemented for dim=2")

    # Fixed corner size
    corner_size = 0.25
    inner_size = 0.125

    # Area of each outer square
    outer_square_area = corner_size * corner_size

    # Area of each inner square
    inner_square_area = inner_size * inner_size

    # Area with R1 reward (outer square minus inner square) × 2 corners
    area_R1 = 2 * (outer_square_area - inner_square_area)

    # Area with R2 reward (inner square) × 2 corners
    area_R2 = 2 * inner_square_area

    # Rest of the space (unit square minus 2 corner squares)
    area_R0 = 1.0 - 2 * outer_square_area

    # Use 1e-9 as baseline instead of R0
    Z = 1e-9 * area_R0 + R1 * area_R1 + R2 * area_R2

    return Z


def reward_edge_boxes(final_states, R0, R1, R2, delta, **kwargs):
    """
    Edge boxes reward: high reward at small nested boxes on North and East edges.

    Structure (for 2D):
    - 2 edge midpoints (North and East) each have nested square boxes
    - Outer box: 2*delta × 2*delta, reward R1
    - Inner box: delta × delta (centered in outer box), reward R2
    - Rest: reward 1e-9 (baseline)

    Edge midpoint positions:
    - East (Right) edge: (1 - delta, 0.5)
    - North (Top) edge: (0.5, 1 - delta)

    Note: Outer box center is positioned at distance delta from the boundary,
    ensuring the outer box (size 2*delta) fits within the [0, 1] × [0, 1] space.
    """
    device = final_states.device
    batch_size = final_states.shape[0]
    dim = final_states.shape[-1]

    if dim != 2:
        raise NotImplementedError("edge_boxes only implemented for dim=2")

    # Box sizes
    outer_size = 2 * delta
    inner_size = delta

    # Initialize with baseline reward (1e-9 instead of 0)
    reward = torch.full((batch_size,), 1e-9, device=device)

    x = final_states[:, 0]  # x coordinate
    y = final_states[:, 1]  # y coordinate

    # Edge box centers - only North and East
    # Position outer box centers so they are delta away from the boundary
    edge_centers = [
        (1.0 - delta, 0.5),  # East (right) edge box center
        (0.5, 1.0 - delta),  # North (top) edge box center
    ]

    for cx, cy in edge_centers:
        # Distance from box center
        dx = torch.abs(x - cx)
        dy = torch.abs(y - cy)

        # Check if inside the outer box (2*delta × 2*delta centered at (cx, cy))
        in_outer = (dx <= outer_size / 2) & (dy <= outer_size / 2)

        # Check if inside the inner box (delta × delta centered at (cx, cy))
        in_inner = (dx <= inner_size / 2) & (dy <= inner_size / 2)

        # Assign rewards: inner overrides outer
        reward[in_outer] = R1
        reward[in_inner] = R2

    return reward


def Z_edge_boxes(dim, R0, R1, R2, delta, **kwargs):
    """Partition function for edge_boxes reward (North and East edges only)."""
    if dim != 2:
        raise NotImplementedError("edge_boxes Z only implemented for dim=2")

    # Box sizes
    outer_size = 2 * delta
    inner_size = delta

    # Areas
    outer_area = outer_size * outer_size
    inner_area = inner_size * inner_size

    # 2 edge boxes (North and East), they don't overlap
    # Area with R1 reward (outer box minus inner box) × 2 boxes
    area_R1 = 2 * (outer_area - inner_area)

    # Area with R2 reward (inner box) × 2 boxes
    area_R2 = 2 * inner_area

    # Rest of the space (unit square minus 2 outer boxes)
    area_R0 = 1.0 - 2 * outer_area

    # Use 1e-9 as baseline, R1 for outer, R2 for inner
    Z = 1e-9 * area_R0 + R1 * area_R1 + R2 * area_R2

    return Z


def reward_edge_boxes_corner_squares(final_states, R0, R1, R2, delta, **kwargs):
    """
    Combined reward: edge boxes (North and East) + corner squares.

    Structure (for 2D):
    - 2 edge midpoints (North and East) each have nested square boxes:
      - Outer box: 2*delta × 2*delta, reward R1
      - Inner box: delta × delta (centered in outer box), reward R2
    - 3 corners (bottom-right, top-left, top-right) each have nested squares:
      - Outer square: 0.25 × 0.25, reward R1
      - Inner square: 0.125 × 0.125 (centered in outer square), reward R2
    - Rest: reward 1e-9 (baseline)

    Edge box centers:
    - East (Right) edge: (1 - delta, 0.5)
    - North (Top) edge: (0.5, 1 - delta)

    Corner positions:
    - Bottom-right: (1, 0)
    - Top-left: (0, 1)
    - Top-right: (1, 1)
    """
    device = final_states.device
    batch_size = final_states.shape[0]
    dim = final_states.shape[-1]

    if dim != 2:
        raise NotImplementedError("edge_boxes_corner_squares only implemented for dim=2")

    # Initialize with baseline reward (1e-9 instead of 0)
    reward = torch.full((batch_size,), 1e-9, device=device)

    x = final_states[:, 0]  # x coordinate
    y = final_states[:, 1]  # y coordinate

    # ----- Edge boxes (North and East) -----
    edge_outer_size = 2 * delta
    edge_inner_size = delta

    edge_centers = [
        (1.0 - delta, 0.5),  # East (right) edge box center
        (0.5, 1.0 - delta),  # North (top) edge box center
    ]

    for cx, cy in edge_centers:
        dx = torch.abs(x - cx)
        dy = torch.abs(y - cy)

        in_outer = (dx <= edge_outer_size / 2) & (dy <= edge_outer_size / 2)
        in_inner = (dx <= edge_inner_size / 2) & (dy <= edge_inner_size / 2)

        reward[in_outer] = R1
        reward[in_inner] = R2

    # ----- Corner squares (3 corners) -----
    corner_outer_size = 0.25
    corner_inner_size = 0.125

    corners = [
        (1.0, 0.0),  # bottom-right
        (0.0, 1.0),  # top-left
        (1.0, 1.0),  # top-right
    ]

    for cx, cy in corners:
        # Distance from corner
        if cx == 0.0:
            dx = x
        else:  # cx == 1.0
            dx = 1.0 - x

        if cy == 0.0:
            dy = y
        else:  # cy == 1.0
            dy = 1.0 - y

        # Check if in outer square
        in_outer = (dx <= corner_outer_size) & (dy <= corner_outer_size)

        # Distance from center of outer square
        dx_from_center = torch.abs(dx - corner_outer_size / 2)
        dy_from_center = torch.abs(dy - corner_outer_size / 2)

        # Inner square is centered in the outer square
        in_inner = (dx_from_center <= corner_inner_size / 2) & (dy_from_center <= corner_inner_size / 2) & in_outer

        # Assign rewards: inner overrides outer
        reward[in_outer] = R1
        reward[in_inner] = R2

    return reward


def Z_edge_boxes_corner_squares(dim, R0, R1, R2, delta, **kwargs):
    """Partition function for combined edge_boxes and corner_squares reward."""
    if dim != 2:
        raise NotImplementedError("edge_boxes_corner_squares Z only implemented for dim=2")

    # ----- Edge boxes contribution -----
    edge_outer_size = 2 * delta
    edge_inner_size = delta
    edge_outer_area = edge_outer_size * edge_outer_size
    edge_inner_area = edge_inner_size * edge_inner_size

    # 2 edge boxes (North and East), they don't overlap
    edge_area_R1 = 2 * (edge_outer_area - edge_inner_area)
    edge_area_R2 = 2 * edge_inner_area

    # ----- Corner squares contribution -----
    corner_outer_size = 0.25
    corner_inner_size = 0.125
    corner_outer_area = corner_outer_size * corner_outer_size
    corner_inner_area = corner_inner_size * corner_inner_size

    # 3 corners (excluding bottom-left)
    corner_area_R1 = 3 * (corner_outer_area - corner_inner_area)
    corner_area_R2 = 3 * corner_inner_area

    # ----- Total areas -----
    total_area_R1 = edge_area_R1 + corner_area_R1
    total_area_R2 = edge_area_R2 + corner_area_R2

    # Rest of the space (unit square minus all boxes and corners)
    area_R0 = 1.0 - 2 * edge_outer_area - 3 * corner_outer_area

    # Use 1e-9 as baseline, R1 for outer regions, R2 for inner regions
    Z = 1e-9 * area_R0 + R1 * total_area_R1 + R2 * total_area_R2

    return Z


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
    'corner_squares': reward_corner_squares,
    'two_corners': reward_two_corners,
    'edge_boxes': reward_edge_boxes,
    'edge_boxes_corner_squares': reward_edge_boxes_corner_squares,
    'debug': reward_debug,
}

Z_FUNCTIONS = {
    'baseline': Z_baseline,
    'ring': Z_ring,
    'angular_ring': Z_angular_ring,
    'multi_ring': Z_multi_ring,
    'curve': Z_curve,
    'gaussian_mixture': Z_gaussian_mixture,
    'corner_squares': Z_corner_squares,
    'two_corners': Z_two_corners,
    'edge_boxes': Z_edge_boxes,
    'edge_boxes_corner_squares': Z_edge_boxes_corner_squares,
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

"""
Configuration file for Continuous GFlowNet training.
Modify the values below to change the training configuration.
"""

# Device configuration
DEVICE = "cuda:2"  # Options: "cuda:0", "cuda:1", "cpu", etc.

# Environment parameters
DIM = 2
DELTA = 0.25
REWARD_DEBUG = False

# Reward configuration
# To change reward-specific parameters (radius, sigma, etc.), edit rewards.py directly
REWARD_TYPE = "corner_small_squares"  # Options: "baseline", "ring", "angular_ring", "multi_ring", "curve", "gaussian_mixture", "corner_squares", "corner_small_squares", "two_corners", "edge_boxes", "edge_boxes_corner_squares", "debug"
R0 = 1e-3 # Baseline reward
R1 = 0.5   # Medium reward (e.g., outer square in corner_squares)
R2 = 10 # High reward (e.g., inner square in corner_squares)

# Model parameters
N_COMPONENTS = 1  # Number of components in Mixture Of Betas
N_COMPONENTS_S0 = 1  # Number of components in Mixture Of Betas for s0
BETA_MIN = 0.1  # Minimum value for the concentration parameters of the Beta distribution
BETA_MAX = 2.0  # Maximum value for the concentration parameters of the Beta distribution
HIDDEN_DIM = 128
N_HIDDEN = 3
UNIFORM_RATIO = 0.0
LOSS = "db"  # Options: "tb", "db"
ALPHA = 1.0  # Options: 0.0, 1.0

# Training parameters
PB = "uniform"  # Options: "learnable", "tied", "uniform"
GAMMA_SCHEDULER = 0.5
SCHEDULER_MILESTONE = 2500
SEED = 0
LR = 1e-3
LR_Z = 1e-3
LR_F = 1e-2
TIE_F = False
BS = 256  # Batch size
N_ITERATIONS = 5000
N_EVALUATION_INTERVAL = 200
N_LOGGING_INTERVAL = 100
N_EVALUATION_TRAJECTORIES = 10000
REPLAY_SIZE = 256
# Logging parameters
NO_PLOT = False
NO_WANDB = False
WANDB_PROJECT = "continuous_gflownets"

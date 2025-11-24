"""
Configuration file for Continuous GFlowNet training.
Modify the values below to change the training configuration.
"""

# Device configuration
DEVICE = "cuda:0"  # Options: "cuda:0", "cuda:1", "cpu", etc.

# Environment parameters
DIM = 2
DELTA = 0.25
ENV_EPSILON = 1e-10
REWARD_DEBUG = False

# Reward configuration
# To change reward-specific parameters (radius, sigma, etc.), edit rewards.py directly
REWARD_TYPE = "baseline"  # Options: "baseline", "ring", "angular_ring", "multi_ring", "curve", "gaussian_mixture"

# Model parameters
N_COMPONENTS = 2  # Number of components in Mixture Of Betas
N_COMPONENTS_S0 = 4  # Number of components in Mixture Of Betas for s0
BETA_MIN = 0.1  # Minimum value for the concentration parameters of the Beta distribution
BETA_MAX = 5.0  # Maximum value for the concentration parameters of the Beta distribution
HIDDEN_DIM = 128
N_HIDDEN = 3

# Training parameters
LOSS = "tb"  # Options: "tb", "db", "modifieddb", "reinforce_tb"
ALPHA = 1.0  # Weight of the reward term in DB
ALPHA_SCHEDULE = 1.0  # Every 1000 iterations, divide alpha by this value
PB = "learnable"  # Options: "learnable", "tied", "uniform"
GAMMA_SCHEDULER = 0.5
SCHEDULER_MILESTONE = 2500
SEED = 0
LR = 1e-3
LR_Z = 1e-3
LR_F = 1e-2
TIE_F = False
BS = 128  # Batch size
N_ITERATIONS = 20000
N_EVALUATION_TRAJECTORIES = 10000

# Logging parameters
NO_PLOT = False
NO_WANDB = False
WANDB_PROJECT = "continuous_gflownets"

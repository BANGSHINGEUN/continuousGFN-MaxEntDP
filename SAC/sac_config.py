"""
Configuration file for SAC-based Continuous GFlowNet training.
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

# Backward policy parameters
PB = "uniform"  # Options: "learnable", "tied", "uniform"

# Training parameters
GAMMA_SCHEDULER = 0.5
SCHEDULER_MILESTONE = 2500
SEED = 0
LR = 1e-3
BS = 128  # Batch size for trajectory sampling
N_ITERATIONS = 20000
N_EVALUATION_TRAJECTORIES = 10000

# SAC-specific parameters
SAC_ALPHA = 1  # SAC temperature parameter
TAU = 0.005  # Target smoothing coefficient
AUTOMATIC_ENTROPY_TUNING = False  # Automatically adjust alpha
TARGET_UPDATE_INTERVAL = 1  # Target network update interval
CRITIC_HIDDEN_SIZE = 256  # Hidden size for critic networks
REPLAY_SIZE = 1000000  # Replay buffer size
SAC_BATCH_SIZE = 256  # Batch size for SAC updates
UPDATES_PER_STEP = 1  # Number of SAC updates per environment step

# Logging parameters
NO_PLOT = False
NO_WANDB = False
WANDB_PROJECT = "continuous_gflownets"

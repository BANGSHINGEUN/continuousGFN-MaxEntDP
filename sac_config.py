"""
Configuration file for SAC-based Continuous GFlowNet training.
SAC-specific parameters only. Common parameters are in config.py.
"""
DEVICE = "cuda:3"  # Options: "cuda:0", "cuda:1", "cpu", etc.

# SAC-specific parameters
TAU = 0.1
UNIFORM_RATIO = 0.1
TARGET_UPDATE_INTERVAL = 5  # Target network update interval
CRITIC_HIDDEN_SIZE = 256  # Hidden size for critic networks
REPLAY_SIZE = 1000000  # Replay buffer size
SAC_BATCH_SIZE = 256  # Batch size for SAC updates
UPDATES_PER_STEP = 5  # Number of SAC updates per environment step

"""
Configuration file for SAC-based Continuous GFlowNet training.
SAC-specific parameters only. Common parameters are in config.py.
"""
DEVICE = "cuda:0"  # Options: "cuda:0", "cuda:1", "cpu", etc.

# SAC-specific parameters
TAU = 0.01
TARGET_UPDATE_INTERVAL = 1  # Target network update interval
POLICY_UPDATE_INTERVAL = 1  # Policy update interval
CRITIC_HIDDEN_SIZE = 256  # Hidden size for critic networks
REPLAY_SIZE = 1000000  # Replay buffer size
SAC_BATCH_SIZE = 256  # Batch size for SAC updates
UPDATES_PER_STEP = 5 # Number of SAC updates per environment step
BIAS_VALUE = 2.0
ALPHA_START_D = 1.5
ALPHA_START_C = 0.6
ALPHA_WARMUP_RATIO = 0.2
ALPHA_RAMP_RATIO = 0.5
WITHOUT_BACKWARD_MODEL = False
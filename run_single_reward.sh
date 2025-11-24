#!/bin/bash
#SBATCH --job-name=cgfn_single
#SBATCH --partition=P2
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Usage: sbatch run_single_reward.sh <reward_type>
# Example: sbatch run_single_reward.sh ring

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source ~/.bashrc
conda activate CGFN

# Change to working directory
cd /home/s2/shingeunbang/continuous-gfn

# Get reward type from argument (default: baseline)
REWARD_TYPE=${1:-baseline}

echo "Running experiment with reward type: ${REWARD_TYPE}"

python main.py \
    --reward_type ${REWARD_TYPE} \
    --device cuda:0 \
    --wandb_project continuous_gflownets_all_rewards

echo "Experiment completed!"

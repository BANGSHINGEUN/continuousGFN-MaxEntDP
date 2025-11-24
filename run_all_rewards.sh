#!/bin/bash
#SBATCH --job-name=cgfn_rewards
#SBATCH --partition=P2
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source ~/.bashrc
conda activate CGFN

# Change to working directory
cd /home/s2/shingeunbang/continuous-gfn

# Define reward types to run
REWARD_TYPES=("baseline" "ring" "angular_ring" "multi_ring" "curve" "gaussian_mixture")

# Run each reward type on a different GPU in parallel
for i in "${!REWARD_TYPES[@]}"; do
    REWARD=${REWARD_TYPES[$i]}
    GPU_ID=$((i % 4))  # Cycle through GPUs 0-3

    echo "Starting ${REWARD} on GPU ${GPU_ID}"

    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
        --reward_type ${REWARD} \
        --device cuda:0 \
        --wandb_project continuous_gflownets_all_rewards \
        > logs/${REWARD}_${SLURM_JOB_ID}.log 2>&1 &

    # Store the process ID
    PIDS[$i]=$!

    # Small delay to avoid race conditions
    sleep 2
done

# Wait for all background jobs to complete
echo "Waiting for all jobs to complete..."
for pid in ${PIDS[@]}; do
    wait $pid
done

echo "All reward experiments completed!"

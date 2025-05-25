#!/bin/bash
#SBATCH --job-name=train_nn          # Job name
#SBATCH --output=logs_mod_train_b/train_nn_%j.out     # Output file (%j = job ID)
#SBATCH --error=logs_mod_train_b/train_nn_%j.err      # Error file (%j = job ID)
#SBATCH --partition=long
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --mem=32gb                    # Memory per node (adjust as needed)
#SBATCH --time=1-23:59:00              # Maximum runtime (hh:mm:ss)

echo "Start loading"
module load python/3.9  # Adjust based on your cluster setup
module load cuda/11.3   # Adjust based on your cluster setup

echo "Start training"
python3 train_nn_mod_with_dropout.py
echo "Stop"

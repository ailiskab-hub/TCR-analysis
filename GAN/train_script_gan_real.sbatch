#!/bin/bash
#SBATCH --job-name=train_GAN
#SBATCH --output=logs_GAN_train_big/train_GAN_%j.out
#SBATCH --error=logs_GAN_train_big/train_GAN_%j.err
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --time=9:59:00

echo "Start loading"
module load python/3.9  # Adjust based on your cluster setup
module load cuda/11.3   # Adjust based on your cluster setup

echo "Start training"
python3 train_GAN_covid.py
echo "Stop"
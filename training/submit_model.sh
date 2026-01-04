#!/bin/bash

#SBATCH --job-name=set_card_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G                      
#SBATCH --time=04:00:00
#SBATCH --partition=gpu                

source /etc/profile
module purge                           
module load conda/Python-ML-2025b-pytorch
module load cuda/12.9
module load nccl/2.27.5-cuda12.9

mkdir -p logs

# 'python -u' is helpful for HPC; it forces "unbuffered" output 
# so you see your logs in real-time instead of waiting for a buffer to fill.
python -u train_model.py
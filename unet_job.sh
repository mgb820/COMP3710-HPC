#!/bin/bash
#SBATCH --job-name=oasis_unet
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=unet_train_%j.out

module load cuda
python3 unet_train.py
#!/bin/bash
#SBATCH --job-name=oasis_unet
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00                 # Reduced to 10 min for faster queue scheduling
#SBATCH --output=unet_demo_%j.out

module load cuda
python3 unet_demo.py
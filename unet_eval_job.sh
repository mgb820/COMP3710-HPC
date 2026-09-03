#!/bin/bash
#SBATCH --job-name=unet_eval
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=unet_eval_%j.out

module load cuda
python3 unet_eval.py
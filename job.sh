#!/bin/bash
#SBATCH --job-name=cuda-check
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
##SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=cuda_check_%j.out
#SBATCH --error=cuda_check_%j.err

# Activate your conda environment
source $HOME/miniconda3/bin/activate
conda activate torch

# Run the script
python test_cuda.py
#!/bin/bash
#SBATCH --job-name=cnn
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cnn_%j.out
#SBATCH --error=cnn_%j.err

# Activate your conda environment
source $HOME/miniconda3/bin/activate
conda activate torch

# Run the script
python Demo_2_Part_3_CNN.py
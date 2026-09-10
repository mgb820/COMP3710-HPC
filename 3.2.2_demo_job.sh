#!/bin/bash
#SBATCH --job-name=cnn
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
##SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cnn_%j.out
#SBATCH --error=cnn_%j.err

# Activate your conda environment
source $HOME/miniconda3/bin/activate
conda activate torch

# Run the script
python 3.2.2_demo.py
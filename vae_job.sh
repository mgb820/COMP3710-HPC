#!/bin/bash
#SBATCH --job-name=oasis_vae
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=vae_%j.out
#SBATCH --error=vae_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

python -u vae.py
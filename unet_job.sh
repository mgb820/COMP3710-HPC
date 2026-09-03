#!/bin/bash
#SBATCH --job-name=unet_oasis
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=unet_train_%j.out

module load cuda
python3 unet_train.py
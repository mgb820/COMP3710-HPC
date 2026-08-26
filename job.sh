cat << 'EOF' > job.sh
#!/bin/bash
#SBATCH --job-name=cifar10_cnn
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cifar10_%j.out
#SBATCH --error=cifar10_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

python Demo_2_Part_3_CNN.py
EOF
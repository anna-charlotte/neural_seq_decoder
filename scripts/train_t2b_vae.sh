#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=lat4
#SBATCH --time=15:00:00
#SBATCH --partition=long
#SBATCH --output=/home/lina4471/willett2023/slurm_out/train_t2b_vae_long%j.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=anna.gerhaher@linacre.ox.ac.uk

echo "Training VAE ..."

module load Anaconda3/2022.05
source activate /data/engs-pnpl/lina4471/venv-willett-pt-3-9--11

module load CMake/3.23.1-GCCcore-11.3.0
module load GCC/11.3.0
module load CUDA/12.0

python -u $DATA/repos/neural_seq_decoder/scripts/train_t2b_vae.py

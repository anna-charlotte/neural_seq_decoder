#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=eval
#SBATCH --mem=400GB
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --output=/home/lina4471/willett2023/rnn/eval_rnn.%j.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=anna.gerhaher@linacre.ox.ac.uk

module load Anaconda3/2022.05
source activate /data/engs-pnpl/lina4471/venv-willett-pt-3-9--11

module load CMake/3.23.1-GCCcore-11.3.0
module load GCC/11.3.0
module load CUDA/12.0

python -u $DATA/repos/neural_seq_decoder/scripts/eval_competition.py --modelPath=/home/lina4471/willett2023/competitionData/model/speechBaseline4


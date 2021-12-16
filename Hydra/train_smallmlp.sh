#!/bin/bash

#SBATCH --job-name=train_smallmlp
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --gpus=1
#SBATCH --partition=pascal_gpu,ampere_gpu
#SBATCH --mail-user=willem.ropke@vub.be
#SBATCH --mail-type=ALL
#SBATCH --output=logs/output-%A.out
#SBATCH --error=logs/err-%A.err

# Load the necessary modules.
module load Python/3.8.6-GCCcore-10.2.0
module load PyTorch/1.9.0-fosscuda-2020b

# Navigate to the job directory.
JOBDIR="/POP-following"
cd $JOBDIR

# Activate the virtual environment.
source venv/bin/activate

# Run the neural network training procedure.
python pop_nn.py \
-dir results/PVI/SDST \
-epochs 5000 \
-batch 512 \
-dropout 0

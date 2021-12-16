#!/bin/bash

#SBATCH --job-name=train_smallmlp
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --gpus=1
#SBATCH --mail-user=willem.ropke@vub.be
#SBATCH --mail-type=ALL
#SBATCH --output=logs/output-%A.out
#SBATCH --error=logs/err-%A.err

# Load the necessary modules.
ml Python/3.9.5-GCCcore-10.3.0
ml PyTorch/1.9.0-fosscuda-2020b

# Navigate to the job directory.
JOBDIR="/POP-following"
cd $JOBDIR

# Activate the virtual environment.
source venv/bin/activate

# Run the neural network training procedure.
./pop_nn.py \
-dir results/PVI/SDST \
-epochs 5000 \
-batch 512 \
-dropout 0

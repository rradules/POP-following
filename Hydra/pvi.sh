#!/bin/bash

#SBATCH --job-name=pvi
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=128gb
#SBATCH --mail-user=willem.ropke@vub.be
#SBATCH --mail-type=ALL
#SBATCH --output=logs/output-%A.out
#SBATCH --error=logs/err-%A.err

# Load the necessary modules.
module load OpenAI-Gym/0.17.1-foss-2019b-Python-3.7.4
module load SciPy-bundle/2021.05-foss-2021a

# Navigate to the job directory.
cd $VSC_HOME/POP-following

# Run the neural network training procedure.
python3 pvi.py \
-novec 15 \
-num_iters 200
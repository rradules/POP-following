#!/bin/bash

#SBATCH --job-name=pql
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64gb
#SBATCH --mail-user=willem.ropke@vub.be
#SBATCH --mail-type=ALL
#SBATCH --output=logs/output-%A.out
#SBATCH --error=logs/err-%A.err

# Load the necessary modules.
module load PyTorch/1.6.0-foss-2019b-Python-3.7.4
module load OpenAI-Gym/0.17.1-foss-2019b-Python-3.7.4
module load scikit-learn/0.22.1-foss-2019b-Python-3.7.4
module load SciPy-bundle/2021.05-foss-2021a

# Install pymoo package not on the cluster
pip uninstall pymoo
pip install --user pymoo

# Navigate to the job directory.
cd $VSC_HOME/POP-following

# Run the neural network training procedure.
python3 pareto_q.py \
-dir results/PQL/SDST \
-novec 10 \
-num_iters 200
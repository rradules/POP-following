#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=8gb
#SBATCH --output=logs/output-%A.out
#SBATCH --error=logs/err-%A.err
#SBATCH --job-name=PVI-SDST
#SBATCH --mail-type=END
#SBATCH --mail-user=roxana@ai.vub.ac.be
#SBATCH --requeue

srun python pvi.py -env 'DeepSeaTreasure-v0' -gamma 0.8 -epsilon 0.05 -decimals 4


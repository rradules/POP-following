#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/output-%A.out
#SBATCH --error=logs/err-%A.err
#SBATCH --job-name=PVI-R2
#SBATCH --mail-type=END
#SBATCH --mail-user=roxana@ai.vub.ac.be
#SBATCH --requeue

srun python pvi.py -env 'RandomMOMDP-v0' -states 20 -obj 2 -act 3 -suc 7 -seed 42 -gamma 0.8 -epsilon 0.1 -decimals 2 -novec 10


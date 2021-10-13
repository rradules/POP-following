#!/bin/bash

#PBS -t 1-1

#SBATCH --time=24:00:00
#SBATCH --mem=4gb
#SBATCH --output=logs/output-%A-%a.out
#SBATCH --error=logs/err-%A-%a.err
#SBATCH --job-name=PVI-MDP
#SBATCH --mail-type=END
#SBATCH --mail-user=roxana@ai.vub.ac.be
#SBATCH --requeue

# Execute the line matching the array index from file params.list:
cmd=`head -${SLURM_ARRAY_TASK_ID} pvirand.list | tail -1`

# Execute the command extracted from the file:
eval $cmd

#!/bin/bash
#SBATCH -p zhanglab.p
#SBATCH -t 3-
#SBATCH -c 2
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 10g
#SBATCH --output tune.%A_%1a.out
#SBATCH --array 0-8
#SBATCH --job-name tune
#SBATCH --mail-type ALL
#SBATCH --mail-user yi.dai0502@gmail.com

declare -i in=$SLURM_ARRAY_TASK_ID%3 ie=$SLURM_ARRAY_TASK_ID/3 i=$SLURM_ARRAY_TASK_ID
limits=(0.0 0.15 0.3 0.45)

python stn_cvx.py tune $1 False $i ${limits[$in]} ${limits[$(expr $in + 1)]} 5 ${limits[$ie]} ${limits[$(expr $ie + 1)]} 5 $2 False



#!/bin/bash
#SBATCH -N 1
#SBATCH -C knl
#SBATCH -q debug
#SBATCH -t 00:15:00


#run the application:
module load python
source activate myenv
srun -n 96 -c 2 python scripts/dataload.py
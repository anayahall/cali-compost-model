#!/bin/bash
#SBATCH -N 1
#SBATCH -C knl
#SBATCH -q debug
#SBATCH -t 00:5:00


#run the application:
module load python
source activate myenv
sruns run -n 96 -c 2 python python scripts/main.py
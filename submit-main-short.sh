#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J compost-singleruncx                              
#SBATCH --mail-user=anayahall@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 08:00:00

#OpenMP settings:
export OMP_NUM_THREADS=64
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the application:
module load python
source activate myenv
srun -n 1 -c 64 --cpu_bind=cores python scripts/main_v2_short.py

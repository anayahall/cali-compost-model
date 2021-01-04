#!/bin/bash
#SBATCH -N 1
#SBATCH -C amd
#SBATCH -q bigmem
#SBATCH -J compost-bigmem
#SBATCH --mail-user=anayahall@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 08:00:00

#OpenMP settings:
export OMP_NUM_THREADS=64
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the application:
module load python
module swap PrgEnv-intel PrgEnv-gnu
source activate mylargemem
srun -n 1 -c 64 --cpu_bind=cores python scripts/main_v2.py

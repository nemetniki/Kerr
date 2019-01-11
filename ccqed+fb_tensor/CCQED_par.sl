#!/bin/bash -e
#SBATCH -J CCQED5_par
#SBATCH -A nesi00480
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --mail-type=all
#SBATCH --mail-user=nnem614@aucklanduni.ac.nz
#SBATCH -o /nesi/project/nesi00480/out/CCQED5_out.txt
#SBATCH -e /nesi/project/nesi00480/err/CCQED5_err.txt
#SBATCH -D /nesi/project/nesi00480/

module load Python/3.6.3-gimkl-2017a

export MKL_NUM_THREADS="N"

srun python MPS_CCQED+fb_par.py 5 .0 .0 .2 .2 .5 0.5 0. -endt 1.1 -L 200 -tol -3 -dt .01 -Nphot 2 -initind1=1

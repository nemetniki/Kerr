#!/bin/bash -e
#SBATCH -J CCQED4_sp
#SBATCH -A nesi00480
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=all
#SBATCH --mail-user=nnem614@aucklanduni.ac.nz
#SBATCH -o /nesi/project/nesi00480/out/CCQED4_oout.txt
#SBATCH -e /nesi/project/nesi00480/err/CCQED4_oerr.txt
#SBATCH -D /nesi/project/nesi00480/

module load Python/3.6.3-gimkl-2017a

srun python MPS_CCQED+fb_sp.py 4 .0 .0 .2 .2 .5 0.5 0. -endt 20. -L 100 -tol -3 -dt .01 -Nphot 2 -initind1=1

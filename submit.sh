#!/usr/bin/bash
#SBATCH --job-name testing
#SBATCH --output run.log
#SBATCH --nodes 1
#SBATCH --ntasks 32
#SBATCH --mem=64G

source ~/.bashrc


REPO_DIR="/users/becker/dopl4670/Research/TDSE_PETSC/cmakebuild/bin"
hostname
pwd

chmod +x $REPO_DIR/simulation.exe

mpirun -np $SLURM_NTASKS $REPO_DIR/simulation.exe $@ >> results.log


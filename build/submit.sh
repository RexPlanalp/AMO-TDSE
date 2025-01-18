#!/usr/bin/bash
#SBATCH --job-name testing
#SBATCH --output run.log
#SBATCH --partition=compute 
#SBATCH --exclude=photon13,photon11
#SBATCH --nodes 1
#SBATCH --ntasks 24
#SBATCH --mem=32G

source ~/spack/share/spack/setup-env.sh

spack load petsc
spack load slepc

REPO_DIR="/home/becker/dopl4670/Research/TDSE/build"
hostname
pwd

mpirun -np $SLURM_NTASKS $REPO_DIR/example >> results.log


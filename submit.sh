#!/usr/bin/bash
#SBATCH --job-name testing
#SBATCH --output run.log
#SBATCH --nodes 1
#SBATCH --ntasks 32
#SBATCH --mem=48G

source ~/spack/share/spack/setup-env.sh

spack load petsc
spack load slepc


REPO_DIR="/users/becker/dopl4670/Research/TDSE_PETSC/cmakebuild/bin"
hostname
pwd

mpirun -np $SLURM_NTASKS $REPO_DIR/simulation.exe $@ >> results.log


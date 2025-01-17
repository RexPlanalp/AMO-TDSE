#!/usr/bin/bash
#SBATCH --job-name testing
#SBATCH --output run.log
#SBATCH --partition=compute --qos=normal
#SBATCH --exclude=photon13,photon11
#SBATCH --nodes 1
#SBATCH --ntasks 24
#SBATCH --mem=32G

source /home/becker/dopl4670/.bashrc

REPO_DIR="/home/becker/dopl4670/Research/TDSE/build"
hostname
pwd

mpirun -np $SLURM_NTASKS $REPO_DIR/example >> results.log


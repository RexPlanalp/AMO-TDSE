#!/usr/bin/bash
#SBATCH --job-name testing
#SBATCH --output testing
#SBATCH --partition=compute --qos=normal
#SBATCH --exclude=photon13,photon11
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=1G

source /home/becker/dopl4670/.bashrc

REPO_DIR="/home/becker/dopl4670/Research/testing"
hostname
pwd

mpirun -np $SLURM_NTASKS $REPO_DIR/example >> results.log


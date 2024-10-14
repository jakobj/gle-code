#!/bin/bash
#SBATCH --job-name=gle-fp16
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=pi_manohar
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
module purge
module load Python/3.10.8-GCCcore-12.2.0
cd /home/jj845/project/gle-code/
source venv/bin/activate
date
OMP_NUM_THREADS=12 time python -m experiments.mnist1d.plastic_e2e --seed=12 --epochs 8 --precision-parameters single --precision-dynamics half --lr 5e-3 --optimizer-step-interval 1 --tau_r-scaling 1.0 --scheduler step
date

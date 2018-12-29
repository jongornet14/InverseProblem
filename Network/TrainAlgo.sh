#!/bin/bash
#SBATCH --nodes=1
#SBATCH --array=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=InverseProblem
#SBATCH --mail-type=START,END
#SBATCH --mail-user=jmg1030@nyu.edu
#SBATCH --output=InverseProblem_%j.out

n=${SLURM_ARRAY_TASK_ID}

module purge

module load anaconda3/5.3.1

python3 InverseProblem.py

echo "done"

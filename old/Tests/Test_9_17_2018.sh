#!/bin/bash
#SBATCH --nodes=1
#SBATCH --array=1-3
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=Test1
#SBATCH --mail-type=END
#SBATCH --mail-user=jmg1030@nyu.edu
#SBATCH --output=Test1_%j.out

n=${SLURM_ARRAY_TASK_ID}

module purge
module load matlab/2018a

echo "Test_9_17_2018(${n})" | matlab

echo "done"

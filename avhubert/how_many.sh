#!/bin/bash

#SBATCH --job-name=How_many
#SBATCH --account=bsc88
#SBATCH -N 1
#SBATCH -c 80                       # Número de CPUs a usar
#SBATCH -t 0-01:00:00              # Tiempo máximo de ejecución
#SBATCH --gres=gpu:4
#SBATCH -q acc_bscls  
#SBATCH --partition=gpu  

#SBATCH --output=/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_ES/out/how_many_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_ES/err/how_many_%j.err

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export SLURM_CPU_BIND=none

python how_many.py
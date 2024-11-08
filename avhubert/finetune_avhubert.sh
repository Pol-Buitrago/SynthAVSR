#!/bin/bash
#SBATCH --job-name=avhubert_finetune
#SBATCH --account=bsc88       
#SBATCH -N 1 
#SBATCH -c 80                           # CPUs por tarea
#SBATCH --gres=gpu:4
#SBATCH --time=1:00:00
#SBATCH -q acc_bscls                    # Quality of Service para GPUs
#SBATCH --partition=gpu
#SBATCH --output=/gpfs/projects/bsc88/speech/research/repos/av_hubert/output/finetune_avhubert_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/research/repos/av_hubert/error/finetune_avhubert_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/bsc/bsc915220/.conda/envs/avhubert

export SLURM_NTASKS=1

python finetune_avhubert.py



#!/bin/bash
#SBATCH --job-name=finetune_SynthAV-CV
#SBATCH --account=bsc88       
#SBATCH -N 1 
#SBATCH -c 80                           # CPUs por tarea
#SBATCH --gres=gpu:4
#SBATCH -t 2-00:00:00 
#SBATCH -q acc_bscls                    # Quality of Service para GPUs
#SBATCH --partition=gpu
#SBATCH --output=/gpfs/projects/bsc88/speech/research/repos/av_hubert/output/finetune_SynthAV-CV_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/research/repos/av_hubert/error/finetune_SynthAV-CV_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/bsc/bsc915220/.conda/envs/avhubert

export SLURM_NTASKS=1

export HYDRA_FULL_ERROR=1

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export SLURM_CPU_BIND=none

python finetune_avhubert_V_Synth.py

#!/bin/bash
#SBATCH --job-name=avhubert_finetune
#SBATCH --account=bsc88       
#SBATCH -N 1 
#SBATCH -c 20                           # CPUs por tarea
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00 
#SBATCH -q acc_bscls                    # Quality of Service para GPUs
#SBATCH --partition=gpu
#SBATCH --output=/gpfs/projects/bsc88/speech/research/repos/av_hubert/output/finetune_avhubert_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/research/repos/av_hubert/error/finetune_avhubert_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/bsc/bsc915220/.conda/envs/avhubert

export SLURM_NTASKS=1
export HYDRA_FULL_ERROR=1

# Configuración de NCCL
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Configuración de PyTorch distribuido
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

python finetune_avhubert_AV.py

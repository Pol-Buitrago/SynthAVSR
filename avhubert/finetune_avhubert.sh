#!/bin/bash
#SBATCH --job-name=avhubert_finetune
#SBATCH --account=bsc88       
#SBATCH --ntasks=4
#SBATCH -c 80                           # CPUs por tarea
#SBATCH --gres=gpu:4
#SBATCH --time=1:00:00
#SBATCH -q acc_bscls                    # Quality of Service para GPUs
#SBATCH --partition=gpu
#SBATCH --output=/gpfs/projects/bsc88/speech/research/repos/av_hubert/output/finetune_avhubert_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/research/repos/av_hubert/error/finetune_avhubert_%j.err


source activate avhubert

#export TORCH_NCCL_P2P_DISABLE=1
#export NCCL_SOCKET_IFNAME=eth0  # Adjust to the correct interface if needed
#export TORCH_NCCL_BLOCKING_WAIT=1
#export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export HYDRA_FULL_ERROR=1 

python finetune_avhubert.py



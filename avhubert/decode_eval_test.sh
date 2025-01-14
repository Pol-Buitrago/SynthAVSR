#!/bin/bash

#SBATCH --job-name=decode
#SBATCH --account=bsc88
#SBATCH -N 1
#SBATCH -c 80                       # Número de CPUs a usar
#SBATCH -t 1-00:00:00              # Tiempo máximo de ejecución
#SBATCH --gres=gpu:4
#SBATCH -q acc_bscls  
#SBATCH --partition=gpu  

#SBATCH --output=/gpfs/projects/bsc88/speech/research/repos/av_hubert/output/decode_test_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/research/repos/av_hubert/error/decode_test_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/bsc/bsc915220/.conda/envs/avhubert

# Definir datasets, modalidades y checkpoints
datasets=("CMU-MOSEAS" "LIP-RTVE" "SpanishCorpus" "Muavic")
modalities=("audio,video" "audio" "video")
checkpoints=("SynthAVSR" "RealAVSR" "MixedAVSR" "SynthAVSR_gan")

# Iterar sobre datasets, modalidades y checkpoints
for dataset in "${datasets[@]}"; do
  for modality in "${modalities[@]}"; do
    for checkpoint in "${checkpoints[@]}"; do
      echo "Ejecutando: Dataset=$dataset, Modalidad=$modality, Checkpoint=$checkpoint"
      python decode.py --dataset "$dataset" --modalities "$modality" --ckpt "$checkpoint" --beam_size 7 --lenpen 0.6
    done
  done
done

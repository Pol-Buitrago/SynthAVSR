#!/bin/bash

#SBATCH --job-name=decode
#SBATCH --account=bsc88
#SBATCH -N 1
#SBATCH -c 80                       # Número de CPUs a usar
#SBATCH -t 2-00:00:00              # Tiempo máximo de ejecución
#SBATCH --gres=gpu:4
#SBATCH -q acc_bscls  
#SBATCH --partition=gpu  

#SBATCH --output=/gpfs/projects/bsc88/speech/research/repos/av_hubert/output/decode_test_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/research/repos/av_hubert/error/decode_test_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/bsc/bsc915220/.conda/envs/avhubert

# Definir parámetros
DATASET="AVCAT"
MODALITY="audio,video"
CHECKPOINT="CAT-AVSR"

# Rango de valores para lenpen y beam_size
LENPEN_VALUES=$(seq -2 0.1 2)
BEAM_SIZE_VALUES=$(seq 10 1 30)

# Iterar sobre los valores de lenpen y beam_size
for lenpen in $LENPEN_VALUES; do
  for beam_size in $BEAM_SIZE_VALUES; do
    echo "Ejecutando: Dataset=$DATASET, Modalidad=$MODALITY, Checkpoint=$CHECKPOINT, Beam size=$beam_size, Lenpen=$lenpen"
    python decode.py \
      --dataset "$DATASET" \
      --modalities "$MODALITY" \
      --ckpt "$CHECKPOINT" \
      --beam_size "$beam_size" \
      --lenpen "$lenpen"
  done
done

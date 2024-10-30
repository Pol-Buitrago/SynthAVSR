#!/bin/bash
#SBATCH --job-name=lrs3_prepare
#SBATCH --account=bsc88
#SBATCH -t 24:00:00                     # Ajusta el tiempo según el tamaño de los datos
#SBATCH -q acc_bscls                    # Quality of Service para GPUs
#SBATCH -c 80                          # Ajusta según disponibilidad
#SBATCH --gres=gpu:4                    # Solicitar dos GPU
#SBATCH --ntasks=1                      # Número de tareas
#SBATCH -N 1                            # Número de nodos
#SBATCH --output=/gpfs/projects/bsc88/speech/research/repos/av_hubert/output/lrs3_prepare_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/research/repos/av_hubert/error/lrs3_prepare_%j.err

# Activar el entorno conda donde tienes las dependencias
source activate avhubert

# Rutas base para los archivos
DATA_DIR="/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data"

# Variables
LRS3_DIR="$DATA_DIR/datasets/dataset_EN/lrs3"                                                   # Directorio donde están almacenados los datos LRS3 (contiene pretrain, trainval y test)
FFMPEG_PATH="/gpfs/home/bsc/bsc915220/.conda/envs/avhubert/bin/ffmpeg"                          # Ruta hacia ffmpeg 
export RANK=0  # Ajusta según necesidades
export N_SHARD=1  # Ajusta según necesidades

# Ejecuta los pasos 1 a 4
for STEP in {1..4}; do
  echo "=== Ejecutando Paso $STEP: Preparación de los datos ==="
  python preparation/lrs3_prepare.py \
    --lrs3 "$LRS3_DIR" \
    --ffmpeg "$FFMPEG_PATH" \
    --rank $RANK \
    --nshard $N_SHARD \
    --step $STEP
done


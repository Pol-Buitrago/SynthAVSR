#!/bin/bash
#SBATCH --job-name=lrs3_duration         # Nombre de la tarea
#SBATCH --account=bsc88       
#SBATCH -t 01:30:00                   # Tiempo (ajustar según el tamaño de los datos)
#SBATCH -q acc_bscls                    # Quality of Service para GPUs
#SBATCH -c 80                           # CPUs por tarea
#SBATCH --gres=gpu:4                    # GPUS
#SBATCH --ntasks=1                      # Número de tareas
#SBATCH -N 1                            # Número de nodos
#SBATCH --output=/gpfs/projects/bsc88/speech/research/repos/av_hubert/output/lrs3_duration_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/research/repos/av_hubert/error/lrs3_duration_%j.err

# Activar el entorno conda donde tienes las dependencias
source activate avhubert

python count_video_duration.py

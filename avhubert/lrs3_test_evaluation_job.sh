#!/bin/bash
#SBATCH --job-name=AVSR_lrs3_test        # Nombre de la tarea
#SBATCH --account=bsc88       
#SBATCH -t 05:00:00                   # Tiempo (ajustar según el tamaño de los datos)
#SBATCH -q acc_bscls                    # Quality of Service para GPUs
#SBATCH -c 80                           # CPUs por tarea
#SBATCH --gres=gpu:4                    # GPUS
#SBATCH --ntasks=1                      # Número de tareas
#SBATCH -N 1                            # Número de nodos
#SBATCH --output=/gpfs/projects/bsc88/speech/research/repos/av_hubert/output/AVSR_lrs3_evaluation_test_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/research/repos/av_hubert/error/AVSR_lrs3_evaluation_test_%j.err

# Activar el entorno conda donde tienes las dependencias
source activate avhubert

python lrs3_test_evaluation.py dummy

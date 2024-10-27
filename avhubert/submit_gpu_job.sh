#!/bin/bash
#SBATCH --account=bsc88
#SBATCH -t 01:00:00                   # Tiempo máximo de ejecución
#SBATCH -q gp_bscls                   # Quality of Service para GPUs
#SBATCH --cpus-per-task=80                # Solicita 16 CPUs por tarea
#SBATCH --ntasks=1                        # Número de tareas

# Cargar módulos necesarios
module load nvidia_hpc_sdk  # Módulo para soporte de GPU
module load Python/3.9.6    # Versi�n de Python
module load CUDA/11.3       # Cargar CUDA si es necesario para PyTorch

# Activar el entorno conda donde tienes las dependencias
source activate avhubert_env

# Ejecutar el script Python
python test.py dummy



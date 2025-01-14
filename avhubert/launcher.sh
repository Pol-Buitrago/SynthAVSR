#!/bin/bash

# Verificar si se pasó un comando como argumento
if [ -z "$1" ]; then
  echo "Por favor, proporciona el comando para ejecutar."
  exit 1
fi

# Crear un nombre de trabajo único para evitar sobrescribir archivos de salida
JOB_NAME="launch_$(date +'%Y%m%d_%H%M%S')"

# Script que será enviado a la cola
cat <<EOT > temp_launcher_script.sh
#!/bin/bash

#SBATCH --job-name=$JOB_NAME
#SBATCH --account=bsc88
#SBATCH -N 1
#SBATCH -c 80                       # Número de CPUs a usar
#SBATCH -t 0-01:00:00              # Tiempo máximo de ejecución
#SBATCH --gres=gpu:4
#SBATCH -q acc_bscls  
#SBATCH --partition=gpu  

#SBATCH --output=/gpfs/projects/bsc88/speech/research/repos/av_hubert/output/$JOB_NAME_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/research/repos/av_hubert/error/$JOB_NAME_%j.err

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate /home/bsc/bsc915220/.conda/envs/avhubert

$1
EOT

# Enviar el script a la cola con sbatch
sbatch temp_launcher_script.sh

# Eliminar el script temporal
rm temp_launcher_script.sh


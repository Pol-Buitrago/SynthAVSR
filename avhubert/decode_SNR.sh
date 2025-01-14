#!/bin/bash

#SBATCH --job-name=decode
#SBATCH --account=bsc88
#SBATCH -N 1
#SBATCH -c 80                       # Número de CPUs a usar
#SBATCH -t 2-00:00:00              # Tiempo máximo de ejecución
#SBATCH --gres=gpu:4
#SBATCH -q acc_bscls  
#SBATCH --partition=gpu  

#SBATCH --output=/gpfs/projects/bsc88/speech/research/repos/av_hubert/output/decode_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/research/repos/av_hubert/error/decode_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/bsc/bsc915220/.conda/envs/avhubert

# Obtener los parámetros pasados al script
MODALITIES=$1
NOISE_TYPE=$2

# Define los valores de los parámetros que deseas iterar
BEAM_SIZES=(20)
LENPENS=(1.5)
SNR_VALUES=$(seq -100 0.5 50)
DATASET="AVCAT"
GEN_SUBSET="test"

# Determinar el tipo de modalidad para nombrar la carpeta y el archivo
if [[ "$MODALITIES" == "audio,video" ]]; then
    MODALITIES_TYPE="AV"
elif [[ "$MODALITIES" == "audio" ]]; then
    MODALITIES_TYPE="A"
elif [[ "$MODALITIES" == "video" ]]; then
    MODALITIES_TYPE="V"
fi
   
# Crea un archivo para almacenar los resultados de WER
RESULTS_DIR="/gpfs/projects/bsc88/speech/research/repos/av_hubert/experiment/decode_wer/SNR/evaluation_cat_full/${NOISE_TYPE}"
mkdir -p $RESULTS_DIR

# Crea un archivo para almacenar los resultados de WER
OUTPUT_FILE="${RESULTS_DIR}/wer_results_with_snr_${MODALITIES_TYPE}.txt"
echo "Beam Size, Lenpen, SNR, WER" > $OUTPUT_FILE

# Iterar sobre todas las combinaciones de beam_size, lenpen y snr
for BEAM_SIZE in "${BEAM_SIZES[@]}"; do
    for LENPEN in "${LENPENS[@]}"; do
        for SNR in $SNR_VALUES; do
            # Llamar al script de Python y capturar el WER
            WER=$(python decode.py --dataset "$DATASET" --ckpt "CAT-AVSR" --modalities "$MODALITIES" --noise_type "$NOISE_TYPE" --snr "$SNR" --beam_size "$BEAM_SIZE" --lenpen "$LENPEN" --gen_subset "$GEN_SUBSET" 2>&1 | grep -m 1 -oP "WER: \K[0-9\.]+")

            # Si se obtiene un valor de WER, guardarlo en el archivo de resultados
            if [[ ! -z "$WER" ]]; then
                echo "$BEAM_SIZE, $LENPEN, $SNR, $WER" >> $OUTPUT_FILE
            else
                echo "Error: No se obtuvo WER para Beam Size $BEAM_SIZE, Lenpen $LENPEN y SNR $SNR" >> $OUTPUT_FILE
            fi
        done
    done
done

echo "¡Ejecución completada! Los resultados de WER se guardaron en $OUTPUT_FILE."

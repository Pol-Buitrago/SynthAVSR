#!/bin/bash
#SBATCH --job-name=iterator1
#SBATCH --account=bsc88       
#SBATCH -N 1 
#SBATCH -c 80                           # CPUs por tarea
#SBATCH --gres=gpu:4
#SBATCH -t 2-00:00:00 
#SBATCH -q acc_bscls                    # Quality of Service para GPUs
#SBATCH --partition=gpu
#SBATCH --output=/gpfs/projects/bsc88/speech/research/repos/av_hubert/output/finetune_iterator_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/research/repos/av_hubert/error/finetune_iterator_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/bsc/bsc915220/.conda/envs/avhubert

# Definir el idioma y otras configuraciones
dataset_language="Spanish"
model_language="Spanish"

# Valores iniciales para lenpen y beam_size
best_wer_test=9999
best_lenpen_test=1.0
best_beam_size_test=20

best_wer_valid=9999
best_lenpen_valid=1.0
best_beam_size_valid=20

# Función para ejecutar el decodificador y obtener solo el valor de WER
get_wer() {
    gen_subset=$1
    beam_size=$2
    lenpen=$3

    # Ejecutar el script de decodificación y capturar el valor de WER en la salida
    wer=$(python decode_avhubert1.py --dataset_language "$dataset_language" --model_language "$model_language" --beam_size $beam_size --lenpen $lenpen --gen_subset $gen_subset 2>&1 | grep -oP 'WER: \K[0-9.]+' | head -n 1)

    # Retornar el valor de WER
    echo $wer
}

# Iterar sobre diferentes valores de beam_size y lenpen
for beam_size in {10..60..10}; do
    for lenpen in $(seq 0.1 0.1 2.0); do
        # Imprimir el estado actual de la iteración
        echo "Probando con Beam Size: $beam_size, Lenpen: $lenpen..."

        # Optimizar para el conjunto de datos 'test'
        echo "Calculando WER para 'test'..."
        wer_test=$(get_wer "test" $beam_size $lenpen)

        # Comparar el WER obtenido para 'test' con el mejor valor actual
        if (( $(echo "$wer_test < $best_wer_test" | bc -l) )); then
            best_wer_test=$wer_test
            best_lenpen_test=$lenpen
            best_beam_size_test=$beam_size
            echo "Nuevo mejor WER para 'test' encontrado: $best_wer_test"
        fi

        # Optimizar para el conjunto de datos 'valid'
        echo "Calculando WER para 'valid'..."
        wer_valid=$(get_wer "valid" $beam_size $lenpen)

        # Comparar el WER obtenido para 'valid' con el mejor valor actual
        if (( $(echo "$wer_valid < $best_wer_valid" | bc -l) )); then
            best_wer_valid=$wer_valid
            best_lenpen_valid=$lenpen
            best_beam_size_valid=$beam_size
            echo "Nuevo mejor WER para 'valid' encontrado: $best_wer_valid"
        fi

        # Imprimir los resultados actuales para este par de parámetros
        echo "Beam Size: $beam_size, Lenpen: $lenpen, WER Test: $wer_test, WER Valid: $wer_valid"
    done
done

# Imprimir los mejores resultados
echo "El mejor WER para el conjunto 'test' es $best_wer_test con Beam Size $best_beam_size_test y Lenpen $best_lenpen_test"
echo "El mejor WER para el conjunto 'valid' es $best_wer_valid con Beam Size $best_beam_size_valid y Lenpen $best_lenpen_valid"

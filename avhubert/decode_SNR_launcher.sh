#!/bin/bash

# Define las combinaciones de modalidades y tipos de ruido
MODALITIES_LIST=("audio")
NOISE_TYPES_LIST=("noise")

# Itera sobre todas las combinaciones de modalidades y tipos de ruido
for MODALITIES in "${MODALITIES_LIST[@]}"; do
    for NOISE_TYPE in "${NOISE_TYPES_LIST[@]}"; do
        # Llama al script original con las combinaciones de parámetros
        sbatch decode_SNR.sh "$MODALITIES" "$NOISE_TYPE"
    done
done

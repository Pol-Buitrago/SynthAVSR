# decode_avhubert.py
"""
Este script permite realizar inferencias con un modelo AV-HuBERT usando los parámetros adecuados 
para un sistema de reconocimiento audiovisual de habla (AVSR). Ajusta el tamaño del haz (beam) y 
la penalización de longitud (lenpen) para optimizar el WER (Word Error Rate).

Asegúrate de tener configuradas las rutas de acceso a los archivos de configuración, el checkpoint, 
y el dataset para que el script funcione correctamente.
"""

import os

# Rutas de acceso configuradas para el entorno
CONFIG_DIR = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/conf"
CHECKPOINT_PATH = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/checkpoints/AVSR_Finetuned_Models/Spanish_ES/best_ckpt.pt"
DATASET_DIR = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_ES/LIP_RTVE/model_data"
RESULTS_PATH = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/experiment/decode/s2s/test"

# Parámetros de decodificación para optimizar el WER
BEAM_SIZE = 20
LENPEN = 1.0

cmd = (
    f"python -B infer_s2s.py --config-dir {CONFIG_DIR} --config-name s2s_decode.yaml "
    f"dataset.gen_subset=test "
    f"common_eval.path={CHECKPOINT_PATH} "
    f"common_eval.results_path={RESULTS_PATH} "
    f"override.modalities=[video,audio] "
    f"common.user_dir=`pwd` "
    f"override.data={DATASET_DIR} "
    f"override.label_dir={DATASET_DIR} "
    f"generation.beam={BEAM_SIZE} "
    f"generation.lenpen={LENPEN} "

)


# Ejecutar el comando
print("Ejecutando inferencia AV-HuBERT con los siguientes parámetros:")
print(f"  Configuración: {CONFIG_DIR}/s2s_decode.yaml")
print(f"  Checkpoint: {CHECKPOINT_PATH}")
print(f"  Dataset: {DATASET_DIR}")
print(f"  Resultados: {RESULTS_PATH}")
print(f"  Beam size: {BEAM_SIZE}")
print(f"  Length penalty: {LENPEN}")
os.system(cmd)

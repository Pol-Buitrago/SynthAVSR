# decode_avhubert.py
"""
Este script permite realizar inferencias con un modelo AV-HuBERT usando los parámetros adecuados 
para un sistema de reconocimiento audiovisual de habla (AVSR). Ajusta el tamaño del haz (beam) y 
la penalización de longitud (lenpen) para optimizar el WER (Word Error Rate).

Asegúrate de tener configuradas las rutas de acceso a los archivos de configuración, el checkpoint, 
y el dataset para que el script funcione correctamente.
"""

import os

# Definir parámetros nuevos
LANG = "es"  # Cambiar según el idioma
GROUP = "test"  # Cambiar según el grupo del dataset
#MODALITIES = "video,audio"  # Cambiar según las modalidades que utilices
MODEL_PATH = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/checkpoints/AVSR_Finetuned_Models/Spanish_ES/best_ckpt.pt"  # Ruta del modelo
TOKENIZER_PATH = "/path/to/tokenizer"  # Ruta del tokenizer BPE
OUT_PATH = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/experiment/decode/s2s/test"  # Ruta de salida

# Rutas de acceso configuradas para el entorno
CONFIG_DIR = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/conf"
DATASET_DIR = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_ES/LIP_RTVE/model_data"
RESULTS_PATH = OUT_PATH  # Usar OUT_PATH para los resultados

# Parámetros de decodificación para optimizar el WER
BEAM_SIZE = 20
LENPEN = 1.0

# Modo de tarea (solo avsr)
TASK = "avsr"
TGT = LANG  # Asignar el idioma de destino directamente
USE_BLEU = False  # No se usa BLEU en AVSR

# Comando para ejecutar la inferencia
cmd = (
    f"python -B infer_s2s.py "
    f"--config-dir {CONFIG_DIR} "
    f"--config-name s2s_decode.yaml "
    f"common.user_dir=`pwd` "
    f"override.modalities=[video,audio] "
    f"dataset.gen_subset={GROUP} "
    f"override.data={DATASET_DIR} "
    f"override.label_dir={DATASET_DIR} "
    f"override.tokenizer_bpe_model={TOKENIZER_PATH} "
    f"generation.beam={BEAM_SIZE} "
    f"generation.lenpen={LENPEN} "
    f"dataset.max_tokens=3000 "
    f"override.labels=[{TGT}] "
    f"override.eval_bleu={USE_BLEU} "
    f"common_eval.path={MODEL_PATH} "
    f"common_eval.results_path={RESULTS_PATH}/{LANG}_{TASK}"
)

# Ejecutar el comando
print("Ejecutando inferencia AV-HuBERT con los siguientes parámetros:")
print(f"  Configuración: {CONFIG_DIR}/s2s_decode.yaml")
print(f"  Checkpoint: {MODEL_PATH}")
print(f"  Dataset: {DATASET_DIR}")
print(f"  Resultados: {RESULTS_PATH}")
print(f"  Beam size: {BEAM_SIZE}")
print(f"  Length penalty: {LENPEN}")
os.system(cmd)

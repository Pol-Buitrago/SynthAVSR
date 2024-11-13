import os
import sys
import argparse
from colorama import Fore, Style, init
from datetime import datetime
import subprocess
import time

"""
python decode_avhubert.py --dataset_language Spanish --model_language Spanish --beam_size 20 --lenpen 1.0 --gen_subset test --use_normalizer
"""

# Inicializar colorama
init(autoreset=True)

# Función para obtener rutas y configuraciones según el idioma
def get_language_paths(dataset_lang, model_lang, gen_subset):
    base_dir = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert"
    model_dir = os.path.join(base_dir, "checkpoints/AVSR_Finetuned_Models")
    data_dir = os.path.join(base_dir, "data/datasets")
    results_base_dir = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/experiment/decode/s2s"

    languages = {
        "Arabic": "Arabic_AR",
        "French": "French_FR",
        "Greek": "Greek_EL",
        "Multilingual": "Multilingual",
        "Russian": "Russian_RU",
        "English": "English_EN",
        "German": "German_DE",
        "Italian": "Italian_IT",
        "Portuguese": "Portuguese_PT",
        "Spanish": "Spanish_ES",
    }

    datasets = {
        "Arabic": "dataset_AR",
        "French": "dataset_FR",
        "Greek": "dataset_EL",
        "Multilingual": "dataset_ML",
        "Russian": "dataset_RU",
        "English": "dataset_EN/lrs3/30h_data",
        "German": "dataset_DE",
        "Italian": "dataset_IT",
        "Portuguese": "dataset_PT",
        "Spanish": "dataset_ES/LIP_RTVE/model_data",
    }

    if dataset_lang not in languages or model_lang not in languages:
        raise ValueError(f"{Fore.RED}Idioma no soportado: {dataset_lang} o {model_lang}")

    results_path = os.path.join(results_base_dir, gen_subset)

    # Generar un identificador único para la ejecución basado en la fecha y hora
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_base_dir, gen_subset, timestamp)

    return {
        "checkpoint_path": os.path.join(model_dir, languages[model_lang], "best_ckpt.pt"),
        "dataset_dir": os.path.join(data_dir, datasets[dataset_lang]),
        "tokenizer_path": os.path.join(model_dir, languages[model_lang], "tokenizer.model"),
        "predictor": "wrd",
        "results_path": results_path
    }

# Argumentos de entrada
parser = argparse.ArgumentParser(description="Script para realizar inferencias con AV-HuBERT")
parser.add_argument("--dataset_language", type=str, required=True,
                    help="Idioma del dataset (e.g., Arabic, French, etc.)")
parser.add_argument("--model_language", type=str, required=True,
                    help="Idioma del modelo (e.g., Arabic, French, etc.)")
parser.add_argument("--beam_size", type=int, default=20,
                    help="Tamaño del beam para decodificación (default: 20)")
parser.add_argument("--lenpen", type=float, default=1.0,
                    help="Penalización de longitud para decodificación (default: 1.0)")
parser.add_argument("--gen_subset", type=str, choices=["test", "valid"], default="test",
                    help="Subconjunto de datos a usar (default: test)")
parser.add_argument("--use_normalizer", action='store_true', default=False, 
                    help="Activar normalización del texto (default: False)")
args = parser.parse_args()

# Configuraciones generales
CONFIG_DIR = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/conf"
USE_BLEU = False

# Obtener rutas y configuraciones
lang_paths = get_language_paths(args.dataset_language, args.model_language, args.gen_subset)
CHECKPOINT_PATH = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/experiment/finetune/20241113_010354/checkpoints/checkpoint_best.pt"
#CHECKPOINT_PATH = lang_paths["checkpoint_path"]
DATASET_DIR = lang_paths["dataset_dir"]
TOKENIZER_PATH = lang_paths["tokenizer_path"]
TGT = lang_paths["predictor"]
RESULTS_PATH = lang_paths["results_path"]

# Comando de inferencia
cmd = (
    f"python -B infer_s2s.py --config-dir {CONFIG_DIR} --config-name s2s_decode.yaml "
    f"dataset.gen_subset={args.gen_subset} "
    f"common_eval.path={CHECKPOINT_PATH} "
    f"common_eval.results_path={RESULTS_PATH} "
    f"override.modalities=[video] "
    f"common.user_dir=`pwd` "
    f"override.data={DATASET_DIR} "
    f"override.label_dir={DATASET_DIR} "
    f"generation.beam={args.beam_size} "
    f"generation.lenpen={args.lenpen} "
    f"override.tokenizer_bpe_model={TOKENIZER_PATH} "
    f"override.labels=[{TGT}] "
    f"override.eval_bleu={USE_BLEU} "
    f"override.use_normalizer={args.use_normalizer}"
)

# Ejecutar el comando con colores solo en los títulos
print(f"{Fore.GREEN}Ejecutando inferencia AV-HuBERT con los siguientes parámetros:")
print(f"  {Fore.CYAN}Configuración:{Style.RESET_ALL} {CONFIG_DIR}/s2s_decode.yaml")
print(f"  {Fore.YELLOW}Checkpoint:{Style.RESET_ALL} {CHECKPOINT_PATH}")
print(f"  {Fore.MAGENTA}Dataset:{Style.RESET_ALL} {DATASET_DIR}")
print(f"  {Fore.BLUE}Resultados:{Style.RESET_ALL} {RESULTS_PATH}\n")
print(f"{Fore.RED}Beam Size:{Style.RESET_ALL} {args.beam_size}")
print(f"{Fore.GREEN}Length Penalty:{Style.RESET_ALL} {args.lenpen}")
print(f"{Fore.CYAN}Dataset Language:{Style.RESET_ALL} {args.dataset_language}")
print(f"{Fore.YELLOW}Model Language:{Style.RESET_ALL} {args.model_language}")
print(f"{Fore.MAGENTA}Gen Subset:{Style.RESET_ALL} {args.gen_subset}")
print(f"{Fore.BLUE}Use Normalizer:{Style.RESET_ALL} {args.use_normalizer}\n")

# Ejecutar el comando con subprocess
process = subprocess.Popen(cmd, shell=True)
process.communicate()  # Espera a que el proceso termine

# Verificar si el archivo de resultados WER existe
results_dir = RESULTS_PATH
wer_file_pattern = "wer.*"  # Buscar archivos que empiecen con "wer."

# Esperar un poco para que el archivo se genere
time.sleep(10)

# Buscar el archivo de WER más reciente
wer_files = [f for f in os.listdir(results_dir) if f.startswith("wer")]
if wer_files:
    # Ordenar los archivos de WER por fecha de modificación
    wer_file = max(wer_files, key=lambda f: os.path.getmtime(os.path.join(results_dir, f)))
    wer_file_path = os.path.join(results_dir, wer_file)

    # Leer el archivo WER y extraer el valor
    with open(wer_file_path, 'r') as f:
        content = f.read()
        # Buscar la línea que contiene el WER
        for line in content.splitlines():
            if "WER:" in line:
                wer_value = float(line.split("WER:")[1].strip())
                print(f"WER: {wer_value}")
                sys.exit(wer_value)  # Salir del script con el valor de WER
else:
    print(f"No se encontró archivo WER en {results_dir}")
    sys.exit(1)  # Salir con un código de error si no se encuentra el archivo


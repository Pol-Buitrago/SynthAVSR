import os
import sys
import argparse
from colorama import Fore, Style, init

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

    return {
        "checkpoint_path": os.path.join(model_dir, languages[model_lang], "best_ckpt.pt"),
        "dataset_dir": os.path.join(data_dir, datasets[dataset_lang]),
        "tokenizer_path": os.path.join(model_dir, languages[model_lang], "tokenizer.model"),
        "predictor": "wrd",
        "results_path": results_path
    }

"""
def extract_wer_from_results(results_path):
    # Buscar los archivos que comienzan con "wer" en el directorio
    for filename in os.listdir(results_path):
        if filename.lower().startswith('wer'):
            result_file_path = os.path.join(results_path, filename)
            break
    else:
        print("No se encontró archivo que empiece con 'wer' en el directorio.")
        return None

    # Leer el archivo para extraer el WER
    with open(result_file_path, 'r') as file:
        lines = file.readlines()
    
    wer_value = None
    for line in lines:
        if line.startswith("WER:"):
            wer_value = line.split(" ")[1].strip()
            break
    
    return wer_value
"""

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
CHECKPOINT_PATH = lang_paths["checkpoint_path"]
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
    f"override.modalities=[video,audio] "
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

# Ejecutar el comando
os.system(cmd)

"""
# Llamar a la función y mostrar el WER
wer_value = extract_wer_from_results(RESULTS_PATH)
if wer_value:
    print(f"\n\n{Fore.GREEN}Word Error Rate (WER): {wer_value}\n\n")  # Color verde para el WER
else:
    print(f"\n\n{Fore.RED}Ha ocurrido algún error en el cálculo del WER\n\n")  # Color rojo para los errores
"""



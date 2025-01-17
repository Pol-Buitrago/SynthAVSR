import os
import argparse
from colorama import Fore, Style, init
from datetime import datetime
import subprocess

# Inicializar colorama
init(autoreset=True)

# =====================
# Funciones Auxiliares
# =====================

def get_dataset_paths(dataset, gen_subset, modalities, ckpt):
    """
    Obtiene las rutas necesarias para ejecutar el modelo según idioma y dataset.
    """
    base_dir = "/gpfs/projects/bsc88/speech/research/repos/av_hubert"
    data_dir = os.path.join(base_dir, "data")
    results_base_dir = os.path.join(base_dir, "experiment/decode/s2s")

    datasets = {
        "LRS3-30h": "datasets/video/lrs3/30h_data",
        "LRS3-433h": "datasets/video/lrs3/433h_data",
        "CMU-MOSEAS": "Manifests/CMU-MOSEAS/data/model_data",
        "LIP-RTVE": "Manifests/LIP-RTVE/data/model_data",
        "SpanishCorpus": "Manifests/SpanishCorpus/data/model_data",
        "Muavic": "Manifests/Muavic/data/model_data",
        "AVCAT":"Manifests/CAT/AVCAT-Benchmark/data/model_data",
    }

    modalities_map = {
        "audio": "A",
        "video": "V",
        "audio,video": "AV"
    }

    if dataset not in datasets:
        raise ValueError(f"{Fore.RED}Dataset no soportado: {dataset}")
    
    if modalities not in modalities_map:
        raise ValueError(f"{Fore.RED}Modalidad no soportada: {modalities}")

    # Generar un identificador único para la ejecución basado en la fecha y hora
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_base_dir, gen_subset, modalities_map[modalities], dataset, ckpt, timestamp)

    return {
        "dataset_dir": os.path.join(data_dir, datasets[dataset]),
        "results_path": results_path
    }

def extract_wer(results_path):
    """
    Extrae el valor de WER desde el fichero wer dentro de la carpeta de resultados.
    """
    wer_file = None
    for root, dirs, files in os.walk(results_path):
        for file in files:
            if file.startswith("wer."):
                wer_file = os.path.join(root, file)
                break
    
    if not wer_file:
        raise FileNotFoundError(f"{Fore.RED}No se encontró el archivo 'wer' en {results_path}")

    # Leer el archivo y extraer el valor de WER
    with open(wer_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("WER:"):
                wer_value = float(line.split()[1])
                return wer_value
    
    raise ValueError(f"{Fore.RED}No se encontró la línea que contiene el valor de WER en {wer_file}")

# =====================
# Configuración
# =====================

CONFIG_DIR = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/conf"
NOISE_BASE_DIR = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/data/musan/tsv"

# =====================
# Argumentos del Script
# =====================

parser = argparse.ArgumentParser(description="Script para realizar inferencias con AV-HuBERT")
parser.add_argument("--dataset", type=str, required=True, help="Dataset (e.g., LIP-RTVE, LRS3-30h, etc.)")
parser.add_argument("--beam_size", type=int, default=20, help="Tamaño del beam (default: 20)")
parser.add_argument("--lenpen", type=float, default=1.0, help="Penalización de longitud (default: 1.0)")
parser.add_argument("--gen_subset", type=str, choices=["test", "valid"], default="test", help="Subconjunto de datos (default: test)")
parser.add_argument("--modalities", type=str, required=True, help="Modalidades a usar (e.g., 'audio', 'video', 'audio,video')")
parser.add_argument("--use_normalizer", action='store_true', default=False, help="Activar normalización de texto (default: False)")
parser.add_argument("--noise_type", type=str, choices=["music", "babble", "noise", "all"], default=None, help="Tipo de ruido para añadir (opcional: music, babble, noise, all)")
parser.add_argument("--noise_prob", type=float, default=1.0, help="Probabilidad de aplicar ruido (default: 1.0)")
parser.add_argument("--snr", type=float, default=None, help="Relación señal-ruido (opcional)")
parser.add_argument("--ckpt", type=str, choices=["SynthAVSR", "RealAVSR", "MixedAVSR", "SynthAVSR_gan", "4hours", "9hours", "19hours", "CAT-AVSR", "Muavic_es", "Muavic_multilingual"], required=True, help="Tipo de checkpoint")

args = parser.parse_args()

# =====================
# Configuración de Rutas
# =====================

# Definición de rutas de checkpoint según modalidad y tipo
BASE_PATH = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/experiment/finetune"
CHECKPOINT_FILE = "checkpoints/checkpoint_best.pt"

CHECKPOINTS = {
    "SynthAVSR_gan": {
        "audio": f"{BASE_PATH}/A/SynthAV-CV_gan/base_vox/20241222_092612/{CHECKPOINT_FILE}",
        "video": f"{BASE_PATH}/V/SynthAV-CV_gan/base_vox/20241222_092637/{CHECKPOINT_FILE}",
        "audio,video": f"{BASE_PATH}/AV/SynthAV-CV_gan/base_vox/20241222_092620/{CHECKPOINT_FILE}"
    },
    "RealAVSR": {
        "audio": f"{BASE_PATH}/A/SpanishCorpus/base_vox/20241221_012618/{CHECKPOINT_FILE}",
        "video": f"{BASE_PATH}/V/SpanishCorpus/base_vox/20241221_215438/{CHECKPOINT_FILE}",
        "audio,video": f"{BASE_PATH}/AV/SpanishCorpus/base_vox/20241221_033100/{CHECKPOINT_FILE}"
    },
    "MixedAVSR": {
        "audio": f"{BASE_PATH}/A/MixCorpus/base_vox/20241222_032650/{CHECKPOINT_FILE}",
        "video": f"{BASE_PATH}/V/MixCorpus/base_vox/20241222_032650/{CHECKPOINT_FILE}",
        "audio,video": f"{BASE_PATH}/AV/MixCorpus/base_vox/20241222_032650/{CHECKPOINT_FILE}"
    },
    "SynthAVSR": {
        "audio": f"{BASE_PATH}/A/SynthAV-CV/base_vox/20241222_092609/{CHECKPOINT_FILE}",
        "video": f"{BASE_PATH}/V/SynthAV-CV/base_vox/20241222_092629/{CHECKPOINT_FILE}",
        "audio,video": f"{BASE_PATH}/AV/SynthAV-CV/base_vox/20241222_092616/{CHECKPOINT_FILE}"
    },
    "4hours": {
        "audio": f"{BASE_PATH}/A/4hours/base_vox/20250104_015720/{CHECKPOINT_FILE}",
        "audio,video": f"{BASE_PATH}/AV/4hours/base_vox/20250104_015833/{CHECKPOINT_FILE}"
    },
    "9hours": {
        "audio": f"{BASE_PATH}/A/LIP-RTVE/base_vox/20250104_020156/{CHECKPOINT_FILE}",
        "audio,video": f"{BASE_PATH}/AV/LIP-RTVE/base_vox/20250104_020149/{CHECKPOINT_FILE}"
    },
    "19hours": {
        "audio": f"{BASE_PATH}/A/19hours/base_vox/20250104_015911/{CHECKPOINT_FILE}",
        "audio,video": f"{BASE_PATH}/AV/19hours/base_vox/20250104_020038/{CHECKPOINT_FILE}"
    },
    "CAT-AVSR": {
        "audio": f"{BASE_PATH}/A/SynthAV-CAT_gan/base_vox/20250104_020219/{CHECKPOINT_FILE}",
        "audio,video": f"{BASE_PATH}/AV/SynthAV-CAT_gan/base_vox/20250104_020220/{CHECKPOINT_FILE}",
        "video": f"{BASE_PATH}/V/SynthAV-CAT_gan/base_vox/20250104_020226/{CHECKPOINT_FILE}"
    },
    "Muavic_es": {
        "audio,video": f"/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/checkpoints/AVSR_Finetuned_Models/Spanish_ES/best_ckpt.pt",
        "audio": f"/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/checkpoints/AVSR_Finetuned_Models/Spanish_ES/best_ckpt.pt"
    },
    "Muavic_multilingual": {
        "audio,video": f"/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/checkpoints/AVSR_Finetuned_Models/Multilingual/best_ckpt.pt",
        "audio": f"/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/checkpoints/AVSR_Finetuned_Models/Multilingual/best_ckpt.pt"
    }
}

# Verificar que el tipo de checkpoint y la modalidad seleccionada tienen una ruta válida
if args.ckpt not in CHECKPOINTS:
    raise ValueError(f"{Fore.RED}Checkpoint no soportado: {args.ckpt}")

CHECKPOINT_PATH = CHECKPOINTS[args.ckpt].get(args.modalities)
if not CHECKPOINT_PATH:
    raise ValueError(f"{Fore.RED}No se encontró el checkpoint para la modalidad {args.modalities} en {args.ckpt}")

# Obtener rutas del dataset
dataset_paths = get_dataset_paths(args.dataset, args.gen_subset, args.modalities, args.ckpt)
DATASET_DIR = dataset_paths["dataset_dir"]
RESULTS_PATH = dataset_paths["results_path"]
TGT = "wrd"

if args.noise_type:
    NOISE_DIR = os.path.join(NOISE_BASE_DIR, args.noise_type)
else:
    NOISE_DIR = None

# =====================
# Construcción del Comando
# =====================

cmd = (
    f"python -B infer_s2s.py --config-dir {CONFIG_DIR} --config-name s2s_decode.yaml "
    f"dataset.gen_subset={args.gen_subset} " 
    f"common_eval.path={CHECKPOINT_PATH} "
    f"common_eval.results_path={RESULTS_PATH} "
    f"override.modalities=[{args.modalities}] "
    f"common.user_dir=/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert "
    f"override.data={DATASET_DIR} "
    f"override.label_dir={DATASET_DIR} "
    f"generation.beam={args.beam_size} "
    f"generation.lenpen={args.lenpen} "
    f"override.labels=[{TGT}] "
    f"override.use_normalizer={args.use_normalizer}"
)

if NOISE_DIR and args.snr is not None:
    cmd += (
        f" +override.noise_wav={NOISE_DIR} "
        f"override.noise_prob={args.noise_prob} "
        f"override.noise_snr={args.snr}"
    )

# =====================
# Ejecución del Comando
# =====================

print(f"{Fore.GREEN}Ejecutando inferencia AV-HuBERT con los siguientes parámetros:")
print(f"  {Fore.CYAN}Configuración:{Style.RESET_ALL} {CONFIG_DIR}/s2s_decode.yaml")
print(f"  {Fore.YELLOW}Checkpoint:{Style.RESET_ALL} {CHECKPOINT_PATH}")
print(f"  {Fore.MAGENTA}Dataset:{Style.RESET_ALL} {DATASET_DIR}")
print(f"  {Fore.BLUE}Resultados:{Style.RESET_ALL} {RESULTS_PATH}\n")
print(f"{Fore.YELLOW}Modalidades:{Style.RESET_ALL} {args.modalities}")
print(f"{Fore.RED}Beam Size:{Style.RESET_ALL} {args.beam_size}")
print(f"{Fore.GREEN}Length Penalty:{Style.RESET_ALL} {args.lenpen}")
print(f"{Fore.CYAN}Dataset:{Style.RESET_ALL} {args.dataset}")
print(f"{Fore.MAGENTA}Gen Subset:{Style.RESET_ALL} {args.gen_subset}")
print(f"{Fore.BLUE}Use Normalizer:{Style.RESET_ALL} {args.use_normalizer}\n")
if NOISE_DIR and args.snr is not None:
    print(f"{Fore.RED}Noise Directory:{Style.RESET_ALL} {NOISE_DIR}")
    print(f"{Fore.GREEN}Noise Probability:{Style.RESET_ALL} {args.noise_prob}")
    print(f"{Fore.BLUE}SNR:{Style.RESET_ALL} {args.snr}")

# Ejecutar el comando
process = subprocess.Popen(cmd, shell=True)
process.communicate()

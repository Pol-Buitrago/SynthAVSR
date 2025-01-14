import os
import cv2
import tempfile
import warnings
from argparse import Namespace, ArgumentParser
import torch
from extract_mouth_ROI import preprocess_video
from video2audio import video2audio
import fairseq
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass.configs import GenerationConfig
from jiwer import wer  # Importar jiwer para calcular el WER
from num2words import num2words
import re
import shutil

def predict(task_type, video_path=None, audio_path=None, transcription_path=None, user_dir="", ckpt_path="", suppress_warnings=True):
    """
    Realiza la predicción utilizando un modelo de reconocimiento de voz audiovisual (AVSR),
    reconocimiento de voz (ASR) o reconocimiento de video (VSR).

    Parámetros:
    - task_type (str): Tipo de tarea, puede ser "AVSR", "ASR", "VSR".
    - video_path (str): Ruta del archivo de video (opcional).
    - audio_path (str): Ruta del archivo de audio (opcional).
    - transcription_path (str): Ruta del archivo de transcripción.
    - user_dir (str): Directorio de usuario opcional para cargar módulos personalizados.
    - ckpt_path (str): Ruta del archivo de checkpoint (opcional).
    - suppress_warnings (bool): Si se deben suprimir advertencias.

    Retorna:
    - reference_text (str): Texto de referencia de la transcripción.
    - hypo (str): Texto de hipótesis generado por el modelo.
    - error_rate (float): Tasa de error de palabras (WER) entre la referencia y la hipótesis.
    """
    
    # Suprimir advertencias opcionalmente
    if suppress_warnings:
        warnings.filterwarnings("ignore")
        torch.serialization.add_safe_globals = lambda *args, **kwargs: None
        torch.nn.utils.parametrizations = torch.nn.utils

    roi_path = None

    if audio_path and not os.path.exists(audio_path):
        video2audio(video_path, audio_path)  # Esto podría necesitar manejar el caso donde video_path es None

    # Definir nombres para el audio y la ROI
    if video_path:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = audio_path or os.path.join(os.path.dirname(video_path), f"{video_name}.wav")
        roi_path = os.path.join(os.path.dirname(video_path), f"{video_name}_roi.mp4")
        
        # Generar ROI y audio si no existen (solo para AVSR y VSR)
        if roi_path and not os.path.exists(roi_path):
            preprocess_video(video_path, roi_path)
    else:
        # Si video_path no se proporciona, asigna nombres por defecto o maneja el caso de ASR
        video_name = os.path.splitext(os.path.basename(audio_path))[0]
        video_path = os.path.join(os.path.dirname(audio_path), f"{video_name}.mp4")
    
    # Calcular número de frames y duración
    num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    duration_ms = int(16_000 * num_frames / 25)

    # Definir directorios y rutas
    base_ckpt_path = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/checkpoints"
    data_dir = tempfile.mkdtemp()

    source_path = '/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_ES/LIP_RTVE/model_data/dict.wrd.txt'
    print(f"Directorio temporal creado en: {data_dir}")
    destination_path = os.path.join(data_dir, 'dict.wrd.txt')
    shutil.copy(source_path, destination_path)

    # Crear archivos TSV y etiquetas según tipo de tarea
    if task_type == "AVSR":
        tsv_cont = ["/\n", f"test-0\t{roi_path}\t{audio_path}\t{num_frames}\t{duration_ms}\n"]
        modalities = ["audio", "video"]
        ckpt_path = os.path.join(base_ckpt_path, "custom_trained_checkpoints", "AVSR_checkpoint_best.pt")

    elif task_type == "ASR":
        tsv_cont = ["/\n", f"test-0\t{None}\t{audio_path}\t{num_frames}\t{duration_ms}\n"]
        modalities = ["audio"]
        ckpt_path = os.path.join(base_ckpt_path, "custom_trained_checkpoints", "ASR_checkpoint_best.pt")

    elif task_type == "VSR":
        tsv_cont = ["/\n", f"test-0\t{roi_path}\t{None}\t{num_frames}\t{duration_ms}\n"]
        modalities = ["video"]
        ckpt_path = os.path.join(base_ckpt_path, "custom_trained_checkpoints", "VSR_checkpoint_best.pt")

    else:
        raise ValueError("Tipo de tarea inválido. Por favor, elige 'AVSR', 'ASR', o 'VSR'.")

    label_cont = ["DUMMY\n"]

    # Guardar archivos TSV y etiquetas
    with open(f"{data_dir}/test.tsv", "w") as fo:
        fo.write("".join(tsv_cont))
    with open(f"{data_dir}/test.wrd", "w") as fo:
        fo.write("".join(label_cont))

    # Importar módulos de usuario
    utils.import_user_module(Namespace(user_dir=user_dir))

    # Configuración de generación
    gen_subset = "test"
    gen_cfg = GenerationConfig(beam=20)
    

    # Cargar modelo y configuraciones
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    models = [model.eval().cuda() for model in models]
    saved_cfg.task.modalities = modalities
    saved_cfg.task.data = data_dir
    saved_cfg.task.label_dir = data_dir
    task = tasks.setup_task(saved_cfg.task)
    #task.load_dataset(gen_subset, task_cfg=saved_cfg.task)
    task.load_dataset(saved_cfg.dataset.gen_subset, task_cfg=task.cfg)

    generator = task.build_generator(models, gen_cfg)

    # Función de decodificación
    def decode_fn(x):
        dictionary = task.target_dictionary
        symbols_ignore = generator.symbols_to_strip_from_output
        symbols_ignore.add(dictionary.pad())
        return task.datasets[gen_subset].label_processors[0].decode(x, symbols_ignore)

    # Procesamiento de predicción y cálculo de WER
    itr = task.get_batch_iterator(dataset=task.dataset(gen_subset)).next_epoch_itr(shuffle=False)
    sample = next(itr)
    sample = utils.move_to_cuda(sample)
    hypos = task.inference_step(generator, models, sample)

    ref = decode_fn(sample['target'][0].int().cpu())
    hypo = hypos[0][0]['tokens'].int().cpu()
    hypo = decode_fn(hypo).upper()

    # Cargar la transcripción de referencia del archivo
    with open(transcription_path, 'r') as f:
        for line in f:
            if line.startswith("Text:"):
                reference_text = line.split("Text:")[1].strip()

    # Convertir números en la hipótesis a palabras
    hypo = convert_numbers_to_words(hypo)
    reference_text = convert_numbers_to_words(reference_text)

    # Calcular WER
    error_rate = wer(reference_text, hypo)

    return reference_text, hypo, error_rate

def convert_numbers_to_words(text):
    # Encuentra todos los números en el texto y los reemplaza por su forma en palabras
    # Cambia guiones por espacios y elimina comas
    return re.sub(r'\b\d+\b', lambda x: re.sub(r',', '', num2words(int(x.group()), lang='en')).replace('-', ' ').upper(), text)



if __name__ == "__main__":
    # Configuración de argumentos de línea de comandos
    parser = ArgumentParser(description="Predicción de AVSR/ASR/VSR")
    parser.add_argument("task_type", type=str, choices=["AVSR", "ASR", "VSR"],
                        help="Tipo de tarea a realizar: AVSR, ASR o VSR.")
    parser.add_argument("--video_path", type=str, required=True, help="Ruta del archivo de video.")
    parser.add_argument("--audio_path", type=str, required=False, help="Ruta del archivo de audio.")
    parser.add_argument("--transcription_path", type=str, required=True,
                        help="Ruta del archivo de transcripción.")
    parser.add_argument("--user_dir", type=str, default="", help="Directorio de usuario para módulos personalizados.")
    parser.add_argument("--ckpt_path", type=str, default="",
                        help="Ruta del archivo de checkpoint (opcional).")
    parser.add_argument("--suppress_warnings", type=bool, default=True,
                        help="Si se deben suprimir advertencias.")

    args = parser.parse_args()

    # Ejecutar predicción y calcular WER
    reference_text, hypo, error_rate = predict(
        args.task_type,
        video_path=args.video_path,
        audio_path=args.audio_path,
        transcription_path=args.transcription_path,
        user_dir=args.user_dir,
        ckpt_path=args.ckpt_path,
        suppress_warnings=args.suppress_warnings
    )

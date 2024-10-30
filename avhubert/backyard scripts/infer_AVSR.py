import os
import cv2
import tempfile
import warnings
from argparse import Namespace
import torch
from extract_mouth_ROI import preprocess_video
from video2audio import video2audio
import fairseq
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.configs import GenerationConfig
from jiwer import wer  # Importar jiwer para calcular el WER

def predict(task_type, video_path=None, audio_path=None, transcription_path=None, user_dir="", ckpt_path="", suppress_warnings=True):
    # Suprimir advertencias opcionalmente
    if suppress_warnings:
        warnings.filterwarnings("ignore")
        torch.serialization.add_safe_globals = lambda *args, **kwargs: None
        torch.nn.utils.parametrizations = torch.nn.utils

    # Verificar y generar ROI y audio
    if video_path and not os.path.exists(video_path):
        origin_clip_path = video_path.replace("roi.mp4", "clip.mp4")
        if os.path.exists(origin_clip_path):
            preprocess_video(origin_clip_path, video_path)
        else:
            raise FileNotFoundError(f"No se encontró ni el archivo ROI ni el clip original: {origin_clip_path}")

    if audio_path and not os.path.exists(audio_path):
        origin_clip_path = audio_path.replace("clip.wav", "clip.mp4")
        if os.path.exists(origin_clip_path):
            video2audio(origin_clip_path, audio_path)
        else:
            raise FileNotFoundError(f"No se encontró ni el archivo de audio ni el clip original: {origin_clip_path}")

    # Calcular número de frames y duración
    num_frames = int(cv2.VideoCapture(video_path.replace("roi.mp4", "clip.mp4")).get(cv2.CAP_PROP_FRAME_COUNT)) if video_path else 0
    duration_ms = int(16_000 * num_frames / 25) if video_path else 0

    # Definir directorios y rutas
    base_ckpt_path = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/checkpoints"
    data_dir = tempfile.mkdtemp()

    # Crear archivos TSV y etiquetas según tipo de tarea
    if task_type == "AVSR":
        tsv_cont = ["/\n", f"test-0\t{video_path}\t{audio_path}\t{num_frames}\t{duration_ms}\n"]
        modalities = ["audio", "video"]
        ckpt_path = os.path.join(base_ckpt_path, "AVSR_Finetuned_Models", "English_EN", "large_noise_pt_noise_ft_433h.pt")
    elif task_type == "ASR":
        tsv_cont = ["/\n", f"test-0\t{None}\t{audio_path}\t{num_frames}\t{duration_ms}\n"]
        modalities = ["audio"]
        ckpt_path = os.path.join(base_ckpt_path, "AVSR_Finetuned_Models", "English_EN", "large_noise_pt_noise_ft_433h.pt")
    elif task_type == "VSR":
        tsv_cont = ["/\n", f"test-0\t{video_path}\t{None}\t{num_frames}\t{duration_ms}\n"]
        modalities = ["video"]
        ckpt_path = os.path.join(base_ckpt_path, "VSR_Finetuned_Models", "self_large_vox_433h.pt")
    else:
        raise ValueError("Invalid task type. Please choose 'AVSR', 'ASR', or 'VSR'.")

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
    task.load_dataset(gen_subset, task_cfg=saved_cfg.task)

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

    # Calcular WER
    error_rate = wer(reference_text, hypo)

    return reference_text, hypo, error_rate

if __name__ == "__main__":
    # Variables de ejemplo
    base_data_dir = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/demos/demo_I"
    mouth_roi_path = os.path.join(base_data_dir, "roi.mp4")
    audio_path = os.path.join(base_data_dir, "clip.wav")
    transcription_path = os.path.join(base_data_dir, "clip.txt")
    
    # Elige el tipo de tarea: AVSR, VSR, ASR o ALL
    task_type = "ALL"  # Cambia a "AVSR", "VSR", "ASR" o "ALL" según la tarea
    user_dir = ""
    
    # Ejecutar predicción y calcular WER
    if task_type == "ALL":
        for mode in ["AVSR", "VSR", "ASR"]:
            print("\n")            
            reference_text, hypo, error_rate = predict(mode, video_path=mouth_roi_path, audio_path=audio_path, transcription_path=transcription_path, user_dir=user_dir)
            print(f"Predicción ({mode}):")
            print(f"  Ground truth: {reference_text}")
            print(f"  Hipótesis: {hypo}")
            print(f"  WER: {error_rate:.2f} - {error_rate * 100}%")
    else:
        print("\n")
        reference_text, hypo, error_rate = predict(task_type, video_path=mouth_roi_path, audio_path=audio_path, transcription_path=transcription_path, user_dir=user_dir)
        print(f"Predicción ({task_type}):")
        print(f"  Ground truth: {reference_text}")
        print(f"  Hipótesis: {hypo}")
        print(f"  WER: {error_rate:.2f} - {error_rate * 100}%")
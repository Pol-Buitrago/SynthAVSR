import os
import cv2
import tempfile
import warnings
from argparse import Namespace
import subprocess
import torch
from extract_mouth_ROI import preprocess_video
from video2audio import video2audio
import fairseq
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.configs import GenerationConfig

def predict(task_type, video_path=None, audio_path=None, user_dir="", ckpt_path="", suppress_warnings=True):
    # Si se especifica suprimir warnings, los suprimimos
    if suppress_warnings:
        warnings.filterwarnings("ignore")
        torch.serialization.add_safe_globals = lambda *args, **kwargs: None  # Suprimir advertencias de torch
        torch.nn.utils.parametrizations = torch.nn.utils  # Evitar el warning de weight_norm

    # Verificar si el archivo ROI ya existe, si no, extraerlo
    if video_path and not os.path.exists(video_path):
        origin_clip_path = video_path.replace("roi.mp4", "clip.mp4")  # Asumimos que el clip original tiene un nombre similar
        if os.path.exists(origin_clip_path):
            preprocess_video(origin_clip_path, video_path)  # Generar ROI
        else:
            raise FileNotFoundError(f"No se encontró ni el archivo ROI ni el clip original: {origin_clip_path}")

    # Verificar si el archivo de audio ya existe, si no, extraerlo
    if audio_path and not os.path.exists(audio_path):
        origin_clip_path = audio_path.replace("clip.wav", "clip.mp4")
        if os.path.exists(origin_clip_path):
            video2audio(origin_clip_path, audio_path)  # Generar archivo de audio
        else:
            raise FileNotFoundError(f"No se encontró ni el archivo de audio ni el clip original: {origin_clip_path}")

    # Determinar el número de frames y la duración del audio para el TSV
    num_frames = int(cv2.VideoCapture(video_path.replace("roi.mp4", "clip.mp4")).get(cv2.CAP_PROP_FRAME_COUNT)) if video_path else 0
    duration_ms = int(16_000 * num_frames / 25) if video_path else 0
    
    # Definir rutas comunes
    base_ckpt_path = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/checkpoints"

    data_dir = tempfile.mkdtemp()
    '''
    # Crear el archivo TSV y sus contenidos dependiendo del tipo de tarea
    if task_type == "AVSR":
        tsv_cont = ["/\n", f"test-0\t{video_path}\t{audio_path}\t{num_frames}\t{duration_ms}\n"]
        modalities = ["audio", "video"]
        ckpt_path = os.path.join(base_ckpt_path, "AVSR_Finetuned_Models", "large_noise_pt_noise_ft_433h.pt")
    elif task_type == "ASR":
        tsv_cont = ["/\n", f"test-0\t{None}\t{audio_path}\t{num_frames}\t{duration_ms}\n"]
        modalities = ["audio"]
        ckpt_path = os.path.join(base_ckpt_path, "AVSR_Finetuned_Models", "large_noise_pt_noise_ft_433h.pt")
    elif task_type == "VSR":
        tsv_cont = ["/\n", f"test-0\t{video_path}\t{None}\t{num_frames}\t{duration_ms}\n"]
        modalities = ["video"]
        ckpt_path = os.path.join(base_ckpt_path, "VSR_Finetuned_Models", "self_large_vox_433h.pt")
    else:
        raise ValueError("Invalid task type. Please choose 'AVSR', 'ASR', or 'VSR'.")
    '''
    label_cont = ["DUMMY\n"]

    tsv_cont = ["/\n", f"test-0\t{video_path}\t{audio_path}\t{num_frames}\t{duration_ms}\n"]
    modalities = ["audio", "video"]
    ckpt_path = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/checkpoints/AVSR_Finetuned_Models/Spanish_ES/best_ckpt.pt"
    #ckpt_path = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/checkpoints/AVSR_Finetuned_Models/English_EN/best_ckpt.pt"

    # Escribir el archivo TSV y el archivo de etiquetas
    with open(f"{data_dir}/test.tsv", "w") as fo:
        fo.write("".join(tsv_cont))
    with open(f"{data_dir}/test.wrd", "w") as fo:
        fo.write("".join(label_cont))
    
    # Importar módulos de usuario
    utils.import_user_module(Namespace(user_dir=user_dir))
    
    # Configurar la generación
    gen_subset = "test"
    gen_cfg = GenerationConfig(beam=20)

    # Cargar el modelo y las configuraciones
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    models = [model.eval().cuda() for model in models]
    
    # Actualizar la configuración con las modalidades y directorios de datos
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

    # Iterar sobre el dataset
    itr = task.get_batch_iterator(dataset=task.dataset(gen_subset)).next_epoch_itr(shuffle=False)
    sample = next(itr)
    sample = utils.move_to_cuda(sample)
    hypos = task.inference_step(generator, models, sample)
    
    ref = decode_fn(sample['target'][0].int().cpu())
    hypo = hypos[0][0]['tokens'].int().cpu()
    hypo = decode_fn(hypo)
    
    return hypo

if __name__ == "__main__":
    # Variables de ejemplo
    base_data_dir = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/demos"
    mouth_roi_path = os.path.join(base_data_dir, "roi.mp4")
    audio_path = os.path.join(base_data_dir, "clip.wav")
    
    # Elige el tipo de tarea: AVSR, VSR, ASR o ALL
    task_type = "AVSR"  # Cambia a "AVSR", "VSR", "ASR" o "ALL" según la tarea
    user_dir = ""
    
    # Ejecuta la predicción según el tipo de tarea
    if task_type == "ALL":
        for mode in ["AVSR", "VSR", "ASR"]:
            hypo = predict(mode, video_path=mouth_roi_path, audio_path=audio_path, user_dir=user_dir)
            print(f"Predicción ({mode}): {hypo}")
    else:
        hypo = predict(task_type, video_path=mouth_roi_path, audio_path=audio_path, user_dir=user_dir)
        print(f"Predicción ({task_type}): {hypo}")


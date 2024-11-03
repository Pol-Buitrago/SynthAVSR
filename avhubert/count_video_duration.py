import os
import cv2
from tqdm import tqdm
from datetime import timedelta

def count_total_duration(folder_path):
    total_duration = timedelta()
    video_count = 0

    # Obtener lista de todos los archivos .mp4 (excluyendo los que terminan en _roi)
    video_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".mp4") and not file.endswith("_roi.mp4"):
                video_files.append(os.path.join(root, file))

    # Usar tqdm para mostrar barra de progreso
    for video_path in tqdm(video_files, desc="Processing videos", unit="video"):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = timedelta(seconds=frame_count / fps)
            total_duration += duration
            video_count += 1
        cap.release()

    # Convertir duración total a hh:mm:ss
    total_seconds = int(total_duration.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Imprimir nombre de la carpeta y resultados en colores
    print(f"\n\033[93mFolder: {folder_path}\033[0m")
    print(f"\033[92mTotal videos: {video_count}\033[0m")
    print(f"\033[96mTotal duration: {hours:02}:{minutes:02}:{seconds:02}\033[0m\n")

# Ejecutar el script con las carpetas especificadas
#count_total_duration("/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_EN/lrs3/pretrain")
#count_total_duration("/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_EN/lrs3/trainval")
#count_total_duration("/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_EN/lrs3/test")
count_total_duration("/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_ES/LIP-RTVE/data/LIP-RTVE/mp4")

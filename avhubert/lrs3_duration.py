import os
import subprocess
from tqdm import tqdm

def get_video_duration(file_path):
    # Usamos ffprobe para obtener la duración del video en segundos
    command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Si ffprobe devuelve un valor válido, lo convertimos a float y lo devolvemos
    try:
        duration = float(result.stdout)
        return duration
    except ValueError:
        return 0  # Si ocurre un error en la conversión, devolvemos 0 segundos

def get_total_video_duration(directory):
    total_duration = 0.0
    video_files = []

    # Recorremos todas las carpetas y archivos en el directorio
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Verificamos si el archivo es un vídeo (por extensión)
            if file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv')):
                file_path = os.path.join(root, file)
                video_files.append(file_path)
    
    # Creamos la barra de progreso usando tqdm
    for file_path in tqdm(video_files, desc="Procesando vídeos", unit="archivo"):
        duration = get_video_duration(file_path)
        total_duration += duration
    
    # Devolvemos la duración total en formato de horas, minutos y segundos
    hours = total_duration // 3600
    minutes = (total_duration % 3600) // 60
    seconds = total_duration % 60
    return int(hours), int(minutes), int(seconds)

# Cambia el path al directorio de tu interés
directory_path = '/gpfs/projects/bsc88/speech/research/repos/av_hubert/data/datasets/video/lrs3/video/short-pretrain'

# Llamamos a la función para obtener la duración total
hours, minutes, seconds = get_total_video_duration(directory_path)

print(f"Duración total de los vídeos: {hours} horas, {minutes} minutos, {seconds} segundos")

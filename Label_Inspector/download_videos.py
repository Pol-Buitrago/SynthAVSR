import yt_dlp
import os

# Función para leer el archivo de URLs
def read_video_urls(file_path):
    with open(file_path, 'r') as file:
        urls = file.readlines()
    # Eliminar saltos de línea
    return [url.strip() for url in urls]

# Función para verificar si el archivo ya ha sido descargado
def is_video_downloaded(video_id, download_dir):
    return os.path.exists(f"{download_dir}/{video_id}.mp4")  # Comprobar si el archivo .mp4 ya existe

# Lista de URLs de vídeos
video_urls = read_video_urls('videos_list.txt')  # Ruta al archivo .txt con las URLs

# Carpeta donde se guardarán los vídeos
download_dir = "full_videos"

# Crear la carpeta si no existe
os.makedirs(download_dir, exist_ok=True)

# Parámetros de descarga
ydl_opts = {
    "format": "bv*[height<=480][ext=mp4]+ba[ext=m4a]/wv*[ext=mp4]+ba[ext=m4a]",  # Preferir MP4 de calidad aceptable
    "outtmpl": f"{download_dir}/%(id)s.%(ext)s",  # Guardar con el ID del vídeo como nombre de archivo en la carpeta 'videos'
    "quiet": True,  # No mostrar mensajes innecesarios
    "merge_output_format": "mp4",  # Especificar que el formato de salida sea MP4
    "postprocessors": [
        {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}  # Convertir al formato MP4 si es necesario
    ],
    "cookies": "cookies.txt",  # Usar un archivo de cookies si es necesario
}

# Descargar los vídeos que no han sido descargados aún
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for url in video_urls:
        video_id = url.split('v=')[-1]  # Obtener el ID del vídeo desde la URL
        if not is_video_downloaded(video_id, download_dir):  # Comprobar si el vídeo ya ha sido descargado
            ydl.download([url])
        else:
            print(f"El vídeo {video_id} ya ha sido descargado.")

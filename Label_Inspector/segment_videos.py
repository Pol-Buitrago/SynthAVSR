import os
from moviepy.editor import VideoFileClip

# Función para cortar el vídeo en fragmentos de 5 segundos
def cut_video_fixed_length(video_path, output_dir, segment_duration=5):
    # Cargar el vídeo
    video = VideoFileClip(video_path)
    
    # Obtener la duración total del vídeo
    video_duration = video.duration
    
    # Crear fragmentos de 5 segundos
    start_time = 0
    fragment_count = 1
    while start_time < video_duration:
        end_time = min(start_time + segment_duration, video_duration)  # Asegurarse de no superar la duración total
        segment = video.subclip(start_time, end_time)
        
        # Guardar el fragmento en la carpeta correspondiente
        segment_filename = os.path.join(output_dir, f"{os.path.basename(video_path).split('.')[0]}_{fragment_count}.mp4")
        segment.write_videofile(segment_filename, codec="libx264", audio_codec="aac")
        
        fragment_count += 1
        start_time = end_time  # Actualizar el tiempo de inicio para el siguiente fragmento

# Función para procesar los vídeos en la carpeta 'videos'
def process_videos(download_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtener la lista de vídeos descargados en la carpeta 'videos'
    video_files = [f for f in os.listdir(download_dir) if f.endswith('.mp4')]  # Filtrar solo archivos mp4
    
    # Procesar cada vídeo
    for video_file in video_files:
        video_path = os.path.join(download_dir, video_file)
        
        # Crear una carpeta específica para cada vídeo
        video_output_dir = os.path.join(output_dir, video_file.split('.')[0])
        os.makedirs(video_output_dir, exist_ok=True)

        print(f"Procesando: {video_path}")
        cut_video_fixed_length(video_path, video_output_dir)

# Directorios de entrada y salida
download_dir = "full_videos"  # Carpeta de vídeos descargados
output_dir = "segments"  # Carpeta para guardar los vídeos cortados

process_videos(download_dir, output_dir)

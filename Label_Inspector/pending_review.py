import os
import shutil

# Rutas de las carpetas
transcriptions_folder = 'transcriptions'  # Cambia esto por la ruta real
recortados_folder = 'revised_videos'          # Cambia esto por la ruta real
pending_folder = 'pending_review'         # Carpeta donde se guardarán los vídeos faltantes

# Crear la carpeta de pending_review si no existe
os.makedirs(pending_folder, exist_ok=True)

# Listar nombres base de transcripciones y vídeos recortados
transcriptions = {os.path.splitext(f)[0] for f in os.listdir(transcriptions_folder) if f.endswith('.txt')}
recortados = {os.path.splitext(f)[0] for f in os.listdir(recortados_folder) if f.endswith('.mp4')}

# Identificar los vídeos que faltan (tienen transcripción pero no vídeo recortado)
pending_videos = transcriptions - recortados

print(f'Vídeos pendientes: {len(pending_videos)}')

# Copiar los vídeos pendientes a la carpeta pending_review
original_videos_folder = 'valid_videos'  # Carpeta con los vídeos originales
for video_name in pending_videos:
    video_path = os.path.join(original_videos_folder, f'{video_name}.mp4')
    if os.path.exists(video_path):
        shutil.copy(video_path, pending_folder)
    else:
        print(f'No se encontró el vídeo original: {video_name}.mp4')

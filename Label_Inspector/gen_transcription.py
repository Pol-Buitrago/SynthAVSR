import os
import whisper
from moviepy.editor import VideoFileClip

# Cargar el modelo Whisper
model = whisper.load_model("turbo")

# Ruta de la carpeta con los vídeos
video_folder = "valid_videos"

# Función para extraer el audio del vídeo
def extract_audio_from_video(video_path):
    video = VideoFileClip(video_path)
    audio_path = video_path.replace(".mp4", ".wav")  # Cambiar extensión a .wav
    video.audio.write_audiofile(audio_path)
    return audio_path

# Procesar todos los vídeos en la carpeta
for video_name in os.listdir(video_folder):
    if video_name.endswith(".mp4"):  # Solo procesar vídeos mp4
        video_path = os.path.join(video_folder, video_name)

        # Extraer el audio
        audio_path = extract_audio_from_video(video_path)

        # Transcribir el audio con Whisper
        result = model.transcribe(audio_path, language="Catalan")

        # Guardar la transcripción en un archivo .txt
        txt_path = video_path.replace(".mp4", ".txt")
        with open(txt_path, "w") as f:
            f.write(result["text"])

        # Eliminar el archivo de audio temporal
        os.remove(audio_path)

        print(f"Transcripción de {video_name} completada.")

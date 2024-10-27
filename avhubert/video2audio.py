import subprocess
import os

def video2audio(video_path, audio_output_path):
    # Comando ffmpeg para extraer audio
    command = [
        "ffmpeg",
        "-i", video_path,        # Entrada: ruta del vídeo
        "-vn",                   # No incluir vídeo
        "-acodec", "pcm_s16le",  # Códec de audio: PCM de 16 bits
        "-ar", "16000",          # Frecuencia de muestreo: 16 kHz
        "-ac", "1",              # Canal: mono
        audio_output_path        # Salida: archivo de audio
    ]

    try:
        # Ejecutar el comando ffmpeg redirigiendo la salida a subprocess.DEVNULL
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Audio extraído exitosamente y guardado en: {audio_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error al extraer audio: {e}")

if __name__ == "__main__":
    # Ruta del vídeo de entrada
    video_path = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/clip.mp4"
    # Ruta del archivo de audio de salida
    audio_output_path = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/clip.wav"
    
    # Convertir vídeo a audio
    video2audio(video_path, audio_output_path)

import os
import numpy as np
from tqdm import tqdm

# Rutas de los ficheros
video_file = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/data/Manifests/CAT/SynthAV-CAT/data/TV3ParlAV/nframes.video"
audio_file = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/data/Manifests/CAT/SynthAV-CAT/data/TV3ParlAV/nframes.audio"
label_file = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/data/Manifests/CAT/SynthAV-CAT/data/TV3ParlAV/label.list"
file_list = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/data/Manifests/CAT/SynthAV-CAT/data/TV3ParlAV/file.list"

# Leer todos los archivos en listas
with open(video_file, 'r') as f:
    video_lines = f.readlines()

with open(audio_file, 'r') as f:
    audio_lines = f.readlines()

with open(label_file, 'r') as f:
    label_lines = f.readlines()

with open(file_list, 'r') as f:
    file_lines = f.readlines()

# Identificar las líneas con 0 en nframes.video
lines_to_remove = [i for i, line in enumerate(video_lines) if line.strip() == '0']

# Filtrar las líneas que no deben eliminarse
video_lines_filtered = [line for i, line in enumerate(video_lines) if i not in lines_to_remove]
audio_lines_filtered = [line for i, line in enumerate(audio_lines) if i not in lines_to_remove]
label_lines_filtered = [line for i, line in enumerate(label_lines) if i not in lines_to_remove]
file_lines_filtered = [line for i, line in enumerate(file_lines) if i not in lines_to_remove]

# Escribir los archivos filtrados de nuevo
with open(video_file, 'w') as f:
    f.writelines(video_lines_filtered)

with open(audio_file, 'w') as f:
    f.writelines(audio_lines_filtered)

with open(label_file, 'w') as f:
    f.writelines(label_lines_filtered)

with open(file_list, 'w') as f:
    f.writelines(file_lines_filtered)

# Barra de progreso
for _ in tqdm(range(100), desc="Eliminando líneas con 0", ncols=100):
    pass

print("Las líneas con 0 han sido eliminadas de todos los archivos.")

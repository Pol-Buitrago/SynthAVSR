import pandas as pd
import os

# Ruta al archivo CSV y la carpeta de vídeos
csv_file_path = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_ES/LIP-RTVE/src/alignments.csv"  
video_folder_path = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_ES/RTVE/20H-RTVEDB2018mp4"

# Cargar el archivo CSV
data = pd.read_csv(csv_file_path)

# Obtener los sourceID únicos y ordenarlos alfabéticamente
unique_source_ids = sorted(data['sourceID'].unique())

# Obtener los nombres de los archivos .mp4 en la carpeta de vídeos
video_files = [f.replace('.mp4', '') for f in os.listdir(video_folder_path) if f.endswith('.mp4')]

# Convertir las listas a conjuntos para facilitar las comparaciones
source_ids_set = set(unique_source_ids)
video_ids_set = set(video_files)

# Identificar los sourceID en el CSV que no tienen archivo .mp4 correspondiente
missing_videos = source_ids_set - video_ids_set
# Identificar los archivos .mp4 que no están en el CSV
extra_videos = video_ids_set - source_ids_set

# Mostrar resultados
print("Source IDs ordenados alfabéticamente:")
for source_id in unique_source_ids:
    print(source_id)

print("\nSource IDs en el CSV sin archivo .mp4 correspondiente:")
for missing in sorted(missing_videos):
    print(missing)

print("\nArchivos .mp4 que no están en el CSV:")
for extra in sorted(extra_videos):
    print(extra)

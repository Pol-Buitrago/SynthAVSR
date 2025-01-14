import numpy as np
import pandas as pd

# Parámetros del área
beam_min, beam_max = 1, 150
lenpen_min, lenpen_max = -10.0, 10
lenpen_step = 0.1

# Leer el archivo existente
file_path = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/experiment/decode_wer/parameter/wer_results_AV.txt"  # Cambia a la ruta de tu archivo
df = pd.read_csv(file_path)

# Generar todas las combinaciones posibles dentro del área, redondeando a 1 decimal
beam_sizes = np.arange(beam_min, beam_max + 1)
lenpens = np.round(np.arange(lenpen_min, lenpen_max + lenpen_step, lenpen_step), 1)
all_combinations = pd.DataFrame(
    [(b, l) for b in beam_sizes for l in lenpens],
    columns=["Beam Size", "Lenpen"]
)

# Identificar las combinaciones faltantes
df = df.rename(columns=lambda x: x.strip())  # Eliminar espacios en columnas
df_present = df[["Beam Size", "Lenpen"]]
missing_combinations = pd.merge(
    all_combinations, df_present, on=["Beam Size", "Lenpen"], how="left", indicator=True
).query('_merge == "left_only"').drop(columns=['_merge'])

# Guardar los valores faltantes en un archivo
missing_file_path = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/experiment/decode_wer/parameter/missing_combinations_AV.txt"
missing_combinations.to_csv(missing_file_path, index=False, header=False, sep=",")
print(f"Combinaciones faltantes guardadas en {missing_file_path}")

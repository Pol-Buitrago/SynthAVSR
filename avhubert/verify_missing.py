import os
import shutil

# Rutas de las carpetas
wav2lip_gan_wav = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_ES/SynthAV-CV/wav2lip_gan/wav"
wav2lip_wav = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_ES/SynthAV-CV/wav2lip/wav"

# Obtener listas de archivos (sin rutas, solo nombres)
gan_files = set(os.listdir(wav2lip_gan_wav))
wav2lip_files = set(os.listdir(wav2lip_wav))

# Encontrar el archivo faltante
missing_files = wav2lip_files - gan_files

if missing_files:
    for file in missing_files:
        # Ruta completa del archivo faltante en wav2lip
        source_path = os.path.join(wav2lip_wav, file)
        # Ruta de destino en wav2lip_gan
        destination_path = os.path.join(wav2lip_gan_wav, file)

        # Copiar el archivo
        shutil.copy(source_path, destination_path)
        print(f"Archivo copiado: {file}")
else:
    print("No se encontraron archivos faltantes.")

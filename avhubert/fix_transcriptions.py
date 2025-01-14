import os
import shutil

# Rutas de las carpetas
wav2lip_gan_transcriptions = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_ES/SynthAV-CV/wav2lip_gan/transcriptions"
wav2lip_transcriptions = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_ES/SynthAV-CV/wav2lip/transcriptions"
source_transcriptions = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_ES/SynthAV-CV/Reprocessed/transcriptions"

# Función para eliminar y copiar la carpeta
def replace_transcriptions(target_path, source_path):
    if os.path.exists(target_path):
        shutil.rmtree(target_path)  # Eliminar la carpeta si existe
        print(f"Eliminada la carpeta: {target_path}")
    shutil.copytree(source_path, target_path)  # Copiar la carpeta desde el origen
    print(f"Copiada la carpeta desde {source_path} a {target_path}")

# Reemplazar en ambas ubicaciones
replace_transcriptions(wav2lip_gan_transcriptions, source_transcriptions)
replace_transcriptions(wav2lip_transcriptions, source_transcriptions)

import os
import pandas as pd

# Directorios base
base_tsv_folder = '/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_ES/CommonVoice/CV-es'
base_mp4_folder = '/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_ES/SynthAV-CV'
transcriptions_folder = '/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_ES/SynthAV-CV/Reprocessed/transcriptions'

# Listas de conjuntos de datos y modelos
datasets = ['train', 'dev', 'test']
models = ['wav2lip_gan', 'wav2lip']

# Iterar sobre los conjuntos de datos y modelos
for dataset in datasets:
    tsv_file = os.path.join(base_tsv_folder, f'{dataset}.tsv')  # Ruta del archivo TSV
    
    # Leer el archivo TSV
    df = pd.read_csv(tsv_file, sep='\t')
    
    # Extraer los nombres de archivo sin la extensión .mp3
    audio_file_names = df['path'].str.replace('.mp3', '', regex=False).tolist()
    
    for model in models:
        # Carpeta de videos .mp4
        mp4_folder = os.path.join(base_mp4_folder, model, 'mp4')
        mp4_files = [f.replace('.mp4', '') for f in os.listdir(mp4_folder) if f.endswith('.mp4')]
        
        # Carpeta de audios .wav
        wav_folder = os.path.join(base_mp4_folder, model, 'wav')
        wav_files = [f.replace('.wav', '') for f in os.listdir(wav_folder) if f.endswith('.wav')]
        
        # Carpeta de transcripciones .txt
        txt_files = [f.replace('.txt', '') for f in os.listdir(transcriptions_folder) if f.endswith('.txt')]
        
        # Contar coincidencias de nombres en .mp4
        matching_mp4_files = [file for file in audio_file_names if file in mp4_files]
        num_mp4_matching = len(matching_mp4_files)
        num_mp4_not_matching = len(audio_file_names) - num_mp4_matching
        
        # Contar coincidencias de nombres en .wav
        matching_wav_files = [file for file in audio_file_names if file in wav_files]
        num_wav_matching = len(matching_wav_files)
        num_wav_not_matching = len(audio_file_names) - num_wav_matching
        
        # Contar coincidencias de nombres en .txt
        matching_txt_files = [file for file in audio_file_names if file in txt_files]
        num_txt_matching = len(matching_txt_files)
        num_txt_not_matching = len(audio_file_names) - num_txt_matching
        
        # Imprimir resultados
        print(f'Dataset: {dataset}, Modelo: {model}')
        print(f'  Número de videos .mp4 ya disponibles: {num_mp4_matching}')
        print(f'  Número de videos .mp4 aún no disponibles: {num_mp4_not_matching}')
        print(f'  Número de audios .wav ya disponibles: {num_wav_matching}')
        print(f'  Número de audios .wav aún no disponibles: {num_wav_not_matching}')
        print(f'  Número de transcripciones .txt ya disponibles: {num_txt_matching}')
        print(f'  Número de transcripciones .txt aún no disponibles: {num_txt_not_matching}')

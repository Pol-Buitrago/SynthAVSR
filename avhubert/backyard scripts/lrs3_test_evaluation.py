import os
from avsr_predictor import predict
from colorama import init, Fore, Style
from jiwer import wer  # Asegúrate de tener jiwer instalado
from tqdm import tqdm  # Asegúrate de tener tqdm instalado para la barra de progreso

# Inicializar colorama
init(autoreset=True)

# Ruta base a los datos
base_data_dir = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/demos/demo_lrs3/test"

# Variables para el WER total
total_error_rate = 0
total_files = 0

# Contar el total de archivos de transcripción .txt
total_txt_files = sum(
    len([name for name in os.listdir(os.path.join(base_data_dir, foldername)) if name.endswith(".txt")])
    for foldername in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, foldername))
)

# Iterar sobre todas las carpetas en la ruta base con barra de progreso
for foldername in os.listdir(base_data_dir):
    folder_path = os.path.join(base_data_dir, foldername)

    # Verificar si es un directorio
    if os.path.isdir(folder_path):
        # Iterar sobre todos los archivos en la carpeta
        for filename in tqdm(os.listdir(folder_path), desc=f"Procesando {foldername}", total=total_txt_files):
            if filename.endswith(".mp4") and not filename.endswith("_roi.mp4"):
                # Construir las rutas de video y transcripción
                video_path = os.path.join(folder_path, filename)
                audio_path = os.path.join(folder_path, filename.replace(".mp4", ".wav"))
                transcription_path = os.path.join(folder_path, filename.replace(".mp4", ".txt"))

                # Comprobar si el archivo de transcripción existe
                if not os.path.exists(transcription_path):
                    print(f"{Fore.RED}Advertencia: No se encontró el archivo de transcripción para {video_path}. Se omitirá.{Style.RESET_ALL}")
                    continue
                
                # Calcular el WER
                reference_text, hypo, error_rate = predict(
                    task_type='ASR',
                    video_path=video_path,
                    audio_path=audio_path,
                    transcription_path=transcription_path
                )

                # Imprimir resultados para cada archivo
                print(f"\n{Fore.CYAN}--- Resultados para {filename} ---{Style.RESET_ALL}")
                print(f"{Fore.GREEN}Referencia: {reference_text}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Hipótesis: {hypo}{Style.RESET_ALL}")
                print(f"{Fore.RED}Tasa de error de palabras (WER): {Style.RESET_ALL}{error_rate:.2f} - {error_rate * 100:.2f}%")

                # Acumular WER total
                total_error_rate += error_rate
                total_files += 1

# Calcular y mostrar el WER total
if total_files > 0:
    average_error_rate = total_error_rate / total_files
    print(f"\n{Fore.MAGENTA}WER total: {average_error_rate:.2f} - {average_error_rate * 100:.2f}% en {total_files} archivos evaluados.{Style.RESET_ALL}")
else:
    print(f"{Fore.RED}No se encontraron archivos de vídeo para evaluar.{Style.RESET_ALL}")

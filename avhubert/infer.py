import os
from avsr_predictor import predict
from colorama import init, Fore, Style

##################################################################################################
########################## Modificar aquí las rutas ##############################################
##################################################################################################
# Ruta base a los datos
base_data_dir = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/demos"

# Definición de las variables de entrada
task_type = 'ALL'  # Cambia a 'ALL' para ejecutar todas las modalidades
video_path = os.path.join(base_data_dir, "clip.mp4")
audio_path = os.path.join(base_data_dir, "clip.wav")
transcription_path = os.path.join(base_data_dir, "clip.txt")
##################################################################################################

# Inicializar colorama
init(autoreset=True)  # Esto restablece el color al predeterminado después de cada impresión

# Si la tarea es 'ALL', iterar sobre las modalidades
if task_type == 'ALL':
    for mode in ['AVSR', 'ASR', 'VSR']:
        reference_text, hypo, error_rate = predict(
            task_type=mode,
            video_path=video_path if mode != 'ASR' else None,
            audio_path=audio_path,
            transcription_path=transcription_path
        )
        
        # Imprimir resultados para cada modalidad
        print(f"\n{Fore.CYAN}--- Resultados para {mode} ---{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Referencia: {Style.RESET_ALL}{reference_text}")
        print(f"{Fore.YELLOW}Hipótesis: {Style.RESET_ALL}{hypo}")
        print(f"{Fore.RED}Tasa de error de palabras (WER): {Style.RESET_ALL}{error_rate:.2f} | {error_rate * 100:.2f}%")
else:
    # Ejecución para un tipo de tarea específico
    reference_text, hypo, error_rate = predict(
        task_type=task_type,
        video_path=video_path,
        audio_path=audio_path,
        transcription_path=transcription_path
    )

    # Imprimir resultados
    print(f"{Fore.CYAN}Referencia: {reference_text}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Hipótesis: {Style.RESET_ALL}{hypo}")
    print(f"{Fore.RED}Tasa de error de palabras (WER): {Style.RESET_ALL}{error_rate:.2f} | {error_rate * 100:.2f}%")

print("\n")

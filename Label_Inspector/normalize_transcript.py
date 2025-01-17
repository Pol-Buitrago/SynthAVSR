import os
import sys
import unicodedata
import re

def normalize_text(text):
    # Eliminar acentos y normalizar caracteres (como 'á' -> 'a')
    text = unicodedata.normalize('NFD', text)  # Normalización de caracteres Unicode

    # Restaurar la 'ñ' y 'Ñ' después de la normalización
    text = re.sub(r'n\u0303', 'ñ', text)  # Buscar 'n' + tilde (~) y reemplazarlo por 'ñ'
    text = re.sub(r'N\u0303', 'Ñ', text)  # Buscar 'N' + tilde (~) y reemplazarlo por 'Ñ'

    # Eliminar marcas de acento en otros caracteres
    text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])  # Eliminar marcas de acento

    # Convertir a minúsculas
    text = text.lower()

    # Eliminar puntuación y caracteres no deseados
    text = re.sub(r'[^\w\s]', '', text)

    return text

def process_files(input_folder, output_folder):
    # Asegurarse de que la carpeta de salida exista
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Recorrer todos los archivos .txt en la carpeta de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):  # Solo procesar archivos .txt
            input_filepath = os.path.join(input_folder, filename)
            
            # Leer el contenido del archivo
            with open(input_filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            # Normalizar el texto
            normalized_text = normalize_text(text)

            # Guardar el texto normalizado en la carpeta de salida
            output_filepath = os.path.join(output_folder, filename)
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(normalized_text)

            print(f"Archivo procesado: {filename}")

if __name__ == '__main__':
    # Carpetas de entrada y salida
    input_folder = 'valid_videos'
    output_folder = 'transcripts'

    # Procesar los archivos
    process_files(input_folder, output_folder)

import unicodedata
import re
import sys

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

if __name__ == '__main__':
    # Leer la transcripción pasada como argumento
    input_text = sys.argv[1]
    normalized_text = normalize_text(input_text)
    
    # Imprimir el texto normalizado
    print(normalized_text)

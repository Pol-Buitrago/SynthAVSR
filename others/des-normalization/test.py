import re
from nltk.corpus import words

# Descargar el diccionario de palabras
import nltk
nltk.download('words')

# Lista de palabras en español (filtrado)
spanish_words = [word.lower() for word in words.words() if re.match(r'^[a-záéíóúñ]+$', word)]

def es_palabra_acentuada(palabra):
    """
    Determina si una palabra necesita tilde según las reglas básicas de acentuación.
    """
    # Verificar si es una palabra que existe en el diccionario
    if palabra in spanish_words:
        # Reglas para palabras agudas, llanas y esdrújulas:
        if palabra[-1] in "nns" or palabra[-2:] in ["ar", "er", "ir"]:
            return False  # Agudas o llanas que terminan en n, s o vocal no llevan tilde
        else:
            return True  # Las demás llevan tilde (palabras esdrújulas)
    return False

def acentuar_palabra(palabra):
    """
    Aplica las reglas de acentuación básica a una palabra.
    """
    palabra = palabra.lower()
    
    # Comprobamos si la palabra está en el diccionario
    if palabra in spanish_words:
        if es_palabra_acentuada(palabra):
            # Aplicamos la tilde en la primera vocal si es necesario
            palabra_acentuada = re.sub(r"([aeiouáéíóú])", r"\1́", palabra, count=1)
            return palabra_acentuada
    return palabra  # Si no es necesario acentuarla, devolvemos la palabra tal cual

def acentuar_texto(texto):
    """
    Acentúa un texto en español sin tildes utilizando el diccionario y reglas.
    """
    palabras = texto.split()
    palabras_acentuadas = [acentuar_palabra(palabra) for palabra in palabras]
    return ' '.join(palabras_acentuadas)

# Ejemplo de uso
texto_sin_tildes = "hola como estas"
texto_acentuado = acentuar_texto(texto_sin_tildes)
print(texto_acentuado)

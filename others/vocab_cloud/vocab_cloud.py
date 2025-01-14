from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Cargar el vocabulario desde el archivo .txt
def cargar_vocabulario(ruta):
    vocabulario = []
    with open(ruta, 'r', encoding='utf-8') as archivo:
        for linea in archivo:
            palabra, frecuencia = linea.split()
            vocabulario.append(palabra)
    return vocabulario

# Generar la nube de palabras
def generar_nube_de_palabras(vocabulario, salida="token_cloud.png"):
    texto = " ".join(vocabulario)
    nube = WordCloud(width=1200, height=300, background_color="white").generate(texto)
    
    # Mostrar la nube de palabras
    plt.figure(figsize=(10, 5))
    plt.imshow(nube, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # Guardar la nube de palabras como imagen
    nube.to_file(salida)
    print(f"Nube de palabras guardada como {salida}")

# Ruta del archivo .txt
archivo_txt = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/data/Manifests/SpanishCorpus/data/model_data/dict.wrd.txt"

# Ejecutar
vocabulario = cargar_vocabulario(archivo_txt)
generar_nube_de_palabras(vocabulario)

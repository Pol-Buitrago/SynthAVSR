# importar las librerías necesarias
import sys
import os

from extract_mouth_ROI import preprocess_video


# Definir las rutas de los archivos
origin_clip_path = "data/clip.mp4"  # Clip original
mouth_roi_path = "data/roi.mp4"  # Destino del ROI generado

# Asegurarse de que las rutas sean correctas
if not all(os.path.exists(path) for path in [origin_clip_path]):
    raise FileNotFoundError("Uno o más archivos no se encuentran en las rutas especificadas.")

# Verificar si el archivo de salida ya existe
if os.path.exists(mouth_roi_path):
    print(f"El archivo de salida {mouth_roi_path} ya existe.")

# Procesar el video y extraer la ROI de la boca
preprocess_video(origin_clip_path, mouth_roi_path)

# Verificar si el archivo de salida se creó correctamente
if os.path.exists(mouth_roi_path):
    print(f"El video ROI de la boca se ha guardado en: {mouth_roi_path}")
else:
    print(f"El video ROI de la boca NO se ha creado: {mouth_roi_path}")

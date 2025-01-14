from PIL import Image

# Lista de imágenes a procesar
image_paths = [
    "video1.png", "video2.png", "video3.png",
    "landmark1.png", "landmark2.png", "landmark3.png"
]

# Recortar imágenes y sobrescribirlas
for image_path in image_paths:
    # Abrir imagen
    image = Image.open(image_path)
    
    # Convertir la imagen a RGBA (si tiene un canal alfa de transparencia)
    image = image.convert("RGBA")
    
    # Recortar solo la parte transparente, manteniendo los bordes
    bbox = image.getbbox()  # Obtener las coordenadas del área no transparente
    image_cropped = image.crop(bbox)  # Recortar la imagen
    
    # Guardar la imagen recortada sobrescribiendo la original
    image_cropped.save(image_path)

print("Las imágenes han sido recortadas manteniendo el borde y sobrescritas.")

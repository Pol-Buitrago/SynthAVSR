import os
import cv2
import numpy as np
import dlib
import shutil

# Cargar el detector de rostros y el predictor de puntos clave
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ruta al modelo de puntos clave

# Función para detectar movimiento en la región de la boca
def detect_mouth_movement(video_path, threshold=100):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print(f"No se pudo leer el video {video_path}")
        return False

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    mouth_movement_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            # Obtener puntos clave faciales
            landmarks = predictor(gray, face)
            mouth_points = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(48, 68)])
            
            # Extraer la región de la boca
            x, y, w, h = cv2.boundingRect(mouth_points)
            mouth_region = gray[y:y+h, x:x+w]

            # Comparar la región con el cuadro anterior
            prev_mouth_region = prev_gray[y:y+h, x:x+w]
            diff = cv2.absdiff(mouth_region, prev_mouth_region)
            movement = np.sum(diff)

            if movement > threshold:
                mouth_movement_detected = True
                break

        prev_gray = gray.copy()

        if mouth_movement_detected:
            break

    cap.release()
    return mouth_movement_detected

# Clasificar los vídeos
def classify_segments(input_dir, valid_dir, invalid_dir, threshold=5):
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(invalid_dir, exist_ok=True)

    total_segments = 0
    valid_count = 0

    for video_folder in os.listdir(input_dir):
        video_folder_path = os.path.join(input_dir, video_folder)
        if not os.path.isdir(video_folder_path):
            continue

        for segment in os.listdir(video_folder_path):
            segment_path = os.path.join(video_folder_path, segment)
            if not segment.endswith('.mp4'):
                continue

            print(f"Procesando: {segment_path}")
            is_valid = detect_mouth_movement(segment_path, threshold)

            # Incrementar los contadores
            total_segments += 1
            if is_valid:
                valid_count += 1

            # Copiar el segmento a la carpeta correspondiente
            output_dir = valid_dir if is_valid else invalid_dir
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy(segment_path, os.path.join(output_dir, segment))

    # Calcular porcentajes
    invalid_count = total_segments - valid_count
    valid_percentage = (valid_count / total_segments) * 100 if total_segments > 0 else 0
    invalid_percentage = (invalid_count / total_segments) * 100 if total_segments > 0 else 0

    print(f"\nClasificación completada:")
    print(f"Total de segmentos: {total_segments}")
    print(f"Válidos: {valid_count} ({valid_percentage:.2f}%)")
    print(f"Inválidos: {invalid_count} ({invalid_percentage:.2f}%)")

# Directorios
input_dir = "segments"
valid_dir = "valis_videos"
invalid_dir = "invalid_videos"

# Clasificar los segmentos
classify_segments(input_dir, valid_dir, invalid_dir, threshold=5)

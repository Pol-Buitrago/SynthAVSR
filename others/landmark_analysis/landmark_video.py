import pickle
import cv2
import numpy as np

# Cargar el archivo .pkl con los landmarks
with open('speaker187_0000.pkl', 'rb') as file:
    landmarks_data = pickle.load(file)

# Cargar el vídeo
cap = cv2.VideoCapture('speaker187_0000.mp4')

# Comprobar que el vídeo se cargó correctamente
if not cap.isOpened():
    print("Error al abrir el vídeo.")
    exit()

# Obtener las propiedades del vídeo original (como el tamaño y el FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Crear el objeto VideoWriter para guardar el nuevo vídeo
out = cv2.VideoWriter('video_con_landmarks.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Si no hay más fotogramas, terminar el bucle
    
    # Obtener los landmarks para el fotograma actual (suponiendo que landmarks_data es un diccionario con el índice del fotograma)
    if frame_idx < len(landmarks_data):
        landmarks = landmarks_data[frame_idx]
        
        # Dibujar los landmarks en el fotograma (supongamos que son puntos (x, y))
        for point in landmarks:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
    
    # Escribir el fotograma con los landmarks en el archivo de salida
    out.write(frame)
    
    # Mostrar el fotograma con los landmarks (opcional)
    cv2.imshow('Video con Landmarks', frame)
    
    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_idx += 1

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()

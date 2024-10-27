import dlib, cv2, os
import numpy as np
import skvideo
import skvideo.io
from tqdm import tqdm
from preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
from base64 import b64encode


def detect_landmark(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def preprocess_video(input_video_path, output_video_path):
    face_predictor_path = "data/misc/shape_predictor_68_face_landmarks.dat"  # Ruta relativa
    mean_face_path = "data/misc/20words_mean_face.npy"  # Ruta relativa

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    STD_SIZE = (256, 256)
    mean_face_landmarks = np.load(mean_face_path)
    stablePntsIDs = [33, 36, 39, 42, 45]

    try:
        # Leer el video
        videogen = skvideo.io.vread(input_video_path)
        frames = np.array([frame for frame in videogen])
        landmarks = []
        
        # Detectar los landmarks en cada frame
        for frame in tqdm(frames):
            landmark = detect_landmark(frame, detector, predictor)
            
            # Verificar si el landmark es válido
            if landmark is None:
                print("Landmark no detectado o inválido en un frame.")
                continue  # Saltar este frame y continuar con el siguiente
            
            landmarks.append(landmark)

        # Después de esta parte, puedes seguir con la interpolación
        preprocessed_landmarks = landmarks_interpolate(landmarks)
        
        # Crear ROI
        rois = crop_patch(input_video_path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, 
                          window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
        
        # Escribir el video procesado
        write_video_ffmpeg(rois, output_video_path, "/gpfs/home/bsc/bsc915220/.conda/envs/avhubert/bin/ffmpeg")
        
        print(f"ROI creada con éxito y guardada en: {output_video_path}")
        
    except Exception as e:
        print(f"Error al crear la ROI: {e}")
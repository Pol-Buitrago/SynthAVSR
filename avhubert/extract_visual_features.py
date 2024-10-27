import os
import cv2
import torch
import utils as avhubert_utils
from argparse import Namespace
import fairseq
from fairseq import checkpoint_utils, utils

def extract_visual_feature(video_path, ckpt_path, user_dir, is_finetune_ckpt=False):
    # Importa los módulos de usuario para AV-HuBERT
    utils.import_user_module(Namespace(user_dir=user_dir))
    
    # Carga el modelo y las configuraciones guardadas
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    
    # Aplica las transformaciones requeridas
    transform = avhubert_utils.Compose([
        avhubert_utils.Normalize(0.0, 255.0),
        avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
        avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std)
    ])
    
    # Carga los frames del vídeo
    frames = avhubert_utils.load_video(video_path)
    print(f"Carga del video {video_path}: dimensiones {frames.shape}")
    
    # Aplica el centro de recorte y normalización
    frames = transform(frames)
    print(f"Centro de recorte del video: dimensiones {frames.shape}")
    
    # Convierte los frames a un tensor y lo mueve a la GPU
    frames = torch.FloatTensor(frames).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
    
    # Obtén el modelo
    model = models[0]
    
    # Verifica si el modelo fue fine-tuneado
    if hasattr(models[0], 'decoder'):
        print("Checkpoint: fine-tuned")
        model = models[0].encoder.w2v_model
    else:
        print("Checkpoint: pre-trained sin fine-tuning")
    
    model.cuda()
    model.eval()

    # Realiza la inferencia sin calcular gradientes
    with torch.no_grad():
        # Especifica output_layer si quieres extraer características de una capa intermedia
        feature, _ = model.extract_finetune(source={'video': frames, 'audio': None}, padding_mask=None, output_layer=None)
        feature = feature.squeeze(dim=0)
    
    print(f"Dimensiones de las características de video: {feature.shape}")
    
    return feature

if __name__ == "__main__":
    # Variables de ejemplo
    mouth_roi_path = "data/roi.mp4"
    ckpt_path = "checkpoints/AVSR_Finetuned_Models/base_noise_pt_noise_ft_433h.pt"
    
    # Obtén el directorio actual (user_dir)
    user_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Extrae las características visuales
    feature = extract_visual_feature(mouth_roi_path, ckpt_path, user_dir)
    
    # Imprime las dimensiones de las características
    print(f"Características extraídas: {feature.shape}")

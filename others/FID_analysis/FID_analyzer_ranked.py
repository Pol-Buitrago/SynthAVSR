import os
from PIL import Image
import cv2  # Usamos OpenCV para procesar los videos
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from tqdm import tqdm  # Importamos tqdm para la barra de progreso
import torch.multiprocessing as mp  # Para paralelización

# Actualiza la función process_video
def process_video(video_path):
    """
    Extract frames from the video and return a list of frames.
    :param video_path: Path to the video file.
    :return: List of frames extracted from the video.
    """
    frames = []
    video_capture = cv2.VideoCapture(video_path)
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)  # Directly append the frame (as a numpy array)
    video_capture.release()
    return frames

# Modifica la función preprocess_image para aceptar directamente el frame
def preprocess_image(frame, resize_to=(299, 299)):
    """
    Preprocess a single frame (numpy array) to be ready for InceptionV3.
    :param frame: Numpy array representing the image frame.
    :param resize_to: Tuple specifying the target size for resizing (default: 299x299).
    :return: Preprocessed image tensor.
    """
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    transform = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.CenterCrop(resize_to),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(pil_image).unsqueeze(0)  # Add batch dimension

# Function to extract features using InceptionV3
def extract_features(model, image_tensors):
    with torch.no_grad():
        features = model(image_tensors).detach().numpy()
    return features

# Function to calculate FID
def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# Function to check if the directory contains valid video or image files
def check_directory(directory_path, allowed_extensions):
    if not os.path.exists(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist.")
        return False
    
    files = [f for f in os.listdir(directory_path) if any(f.lower().endswith(ext) for ext in allowed_extensions)]
    
    if len(files) == 0:
        print(f"Error: No valid files found in '{directory_path}'.")
        return False
    
    print(f"Found {len(files)} valid files in '{directory_path}'.")
    return True

# Function to process shard of videos or images with progress tracking
def process_shard(shard_idx, num_shards, video_files, image_files, model, real_video_dir, generated_image_dir):
    real_features = []
    generated_features = []

    # Determine the range of files for this shard
    shard_video_files = video_files[shard_idx::num_shards]
    shard_image_files = image_files[shard_idx::num_shards]
    
    # Add tqdm to show progress for processing video files
    print(f"Processing shard {shard_idx + 1} of {num_shards}...")
    
    # Process real videos (extract frames)
    for filename in tqdm(shard_video_files, desc=f"Processing videos (Shard {shard_idx + 1})", position=0, leave=False):
        video_path = os.path.join(real_video_dir, filename)
        frames = process_video(video_path)
        for frame in frames:
            image_tensor = preprocess_image(frame)
            features = extract_features(model, image_tensor)
            real_features.append(features)

    # Process generated images
    for filename in tqdm(shard_image_files, desc=f"Processing images (Shard {shard_idx + 1})", position=0, leave=False):
        image_path = os.path.join(generated_image_dir, filename)
        image_tensor = preprocess_image(image_path)
        features = extract_features(model, image_tensor)
        generated_features.append(features)

    return real_features, generated_features

# Main pipeline with progress tracking
def fid_pipeline(real_video_dir, generated_image_dir, num_shards=1):
    if not check_directory(real_video_dir, ['.mp4', '.avi', '.mov']):
        return  # Exit if real videos directory is not valid
    if not check_directory(generated_image_dir, ['.jpg', '.jpeg', '.png']):
        return  # Exit if generated images directory is not valid
    
    # Load pretrained InceptionV3 model
    model = inception_v3(weights='IMAGENET1K_V1', transform_input=False)
    model.fc = torch.nn.Identity()  # Remove the final classification layer
    model.eval()

    # Get list of video and image files
    video_files = [f for f in os.listdir(real_video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    image_files = [f for f in os.listdir(generated_image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Create multiprocessing pool
    with mp.Pool(processes=num_shards) as pool:
        results = [
            pool.apply_async(process_shard, (shard_idx, num_shards, video_files, image_files, model, real_video_dir, generated_image_dir))
            for shard_idx in range(num_shards)
        ]
        
        # Collect the results
        real_features = []
        generated_features = []
        for result in tqdm(results, desc="Processing Shards", position=0, leave=False):
            shard_real_features, shard_generated_features = result.get()
            real_features.extend(shard_real_features)
            generated_features.extend(shard_generated_features)

    # Stack features and calculate statistics
    real_features = np.vstack(real_features)
    generated_features = np.vstack(generated_features)

    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)

    # Calculate FID
    fid_score = calculate_fid(mu1, sigma1, mu2, sigma2)
    return fid_score

# Example usage
if __name__ == "__main__":
    real_videos_path = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/data/datasets/video/SynthAV-CV/wav2lip_gan/mp4"
    generated_images_path = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/data/datasets/img/ffhq-dataset/clean-images1024x1024"
    num_shards = 100  # Número de shards

    fid = fid_pipeline(real_videos_path, generated_images_path, num_shards=num_shards)
    if fid is not None:
        print(f"FID score: {fid:.2f}")

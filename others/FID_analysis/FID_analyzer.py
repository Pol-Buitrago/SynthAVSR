import os
from PIL import Image
import cv2  # Usamos OpenCV para procesar los videos
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from tqdm import tqdm  # Importamos tqdm para la barra de progreso
import random

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
        # Convert the frame to a PIL image and add to the list
        frames.append(frame)  # Directly append the frame (as a numpy array)
    video_capture.release()
    return frames

def preprocess_image(input_data, resize_to=(299, 299)):
    """
    Preprocess an image or frame to be ready for InceptionV3.
    :param input_data: Either a numpy array (frame) or a file path to an image.
    :param resize_to: Tuple specifying the target size for resizing (default: 299x299).
    :return: Preprocessed image tensor.
    """
    # Check if input_data is a numpy array (video frame)
    if isinstance(input_data, np.ndarray):
        # Convert numpy array (frame) to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB))
    else:
        # Otherwise, assume it's a file path and open the image
        pil_image = Image.open(input_data).convert("RGB")

    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.CenterCrop(resize_to),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(pil_image).unsqueeze(0)  # Add batch dimension


# Function to extract features using InceptionV3
def extract_features(model, image_tensors):
    """
    Extract features using the InceptionV3 model.
    :param model: Pretrained InceptionV3 model.
    :param image_tensors: Batch of preprocessed image tensors.
    :return: Numpy array of features.
    """
    with torch.no_grad():
        features = model(image_tensors).detach().numpy()
    return features

# Function to calculate FID
def calculate_fid(mu1, sigma1, mu2, sigma2):
    """
    Calculate the Frechet Inception Distance (FID).
    :param mu1: Mean of the real data features.
    :param sigma1: Covariance matrix of the real data features.
    :param mu2: Mean of the generated data features.
    :param sigma2: Covariance matrix of the generated data features.
    :return: FID score.
    """
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# Function to check if the directory contains valid video or image files
def check_directory(directory_path, allowed_extensions):
    """
    Verifies if the directory contains any valid video or image files.
    :param directory_path: Path to the directory.
    :param allowed_extensions: List of allowed file extensions (e.g., for images or videos).
    :return: True if directory contains valid files, False otherwise.
    """
    if not os.path.exists(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist.")
        return False
    
    files = [f for f in os.listdir(directory_path) if any(f.lower().endswith(ext) for ext in allowed_extensions)]
    
    if len(files) == 0:
        print(f"Error: No valid files found in '{directory_path}'.")
        return False
    
    print(f"Found {len(files)} valid files in '{directory_path}'.")
    return True

def save_example(image, output_path):
    """
    Save an image tensor or numpy array to a specified file path.
    :param image: Image tensor or numpy array.
    :param output_path: Path to save the image.
    """
    if isinstance(image, torch.Tensor):
        # Convert tensor to PIL Image
        image = transforms.ToPILImage()(image.squeeze(0))  # Remove batch dimension
    elif isinstance(image, np.ndarray):
        # Convert numpy array (frame) to PIL Image
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image.save(output_path)

def fid_pipeline(real_video_dir, generated_image_dir, video_sample_ratio=0.00001, image_sample_ratio=0.005):
    """
    Calculate the FID score between real video dataset and generated image dataset.
    :param real_video_dir: Path to the directory of real videos.
    :param generated_image_dir: Path to the directory of generated images.
    :param video_sample_ratio: Proportion of videos to process (default: 0.00001).
    :param image_sample_ratio: Proportion of images to process (default: 5%).
    :return: FID score.
    """
    # Check if both directories contain valid video/image files
    if not check_directory(real_video_dir, ['.mp4', '.avi', '.mov']):
        return  # Exit if real videos directory is not valid
    if not check_directory(generated_image_dir, ['.jpg', '.jpeg', '.png']):
        return  # Exit if generated images directory is not valid

    # Load pretrained InceptionV3 model
    model = inception_v3(weights='IMAGENET1K_V1', transform_input=False)
    model.fc = torch.nn.Identity()  # Remove the final classification layer
    model.eval()

    # Initialize lists to store features
    real_features = []
    generated_features = []

    # Process real videos (extract frames)
    print("Processing real videos (or frames)...")
    video_files = [f for f in os.listdir(real_video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    # Select a random sample of videos
    sample_size = max(1, int(len(video_files) * video_sample_ratio))  # Ensure at least 1 video is processed
    video_files_sample = random.sample(video_files, sample_size)
    print(f"Selected {len(video_files_sample)} videos out of {len(video_files)} for processing.")
    
    for filename in tqdm(video_files_sample, desc="Real Videos", unit="video"):
        video_path = os.path.join(real_video_dir, filename)
        frames = process_video(video_path)
        for idx, frame in enumerate(frames):
            image_tensor = preprocess_image(frame)
            if idx == 0:  # Save the first frame as an example
                save_example(image_tensor, f"/gpfs/projects/bsc88/speech/research/repos/av_hubert/others/FID_analysis/frame_{filename}_example.jpg")

            features = extract_features(model, image_tensor)
            real_features.append(features)

    # Process generated images
    print("Processing generated images...")
    image_files = [f for f in os.listdir(generated_image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Select a random sample of images
    sample_size = max(1, int(len(image_files) * image_sample_ratio))  # Ensure at least 1 image is processed
    image_files_sample = random.sample(image_files, sample_size)
    print(f"Selected {len(image_files_sample)} images out of {len(image_files)} for processing.")
    
    for filename in tqdm(image_files_sample, desc="Generated Images", unit="image"):
        image_path = os.path.join(generated_image_dir, filename)
        image_tensor = preprocess_image(image_path)
        features = extract_features(model, image_tensor)
        generated_features.append(features)

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
    real_videos_path = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/data/datasets/video/SynthAV-CV/wav2lip_gan/mp4"  # Real videos
    generated_images_path = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/data/datasets/img/ffhq-dataset/clean-images1024x1024"  # Generated images

    fid = fid_pipeline(real_videos_path, generated_images_path)
    if fid is not None:
        print(f"FID score: {fid:.2f}")

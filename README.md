
# Spanish-AVSR (Audiovisual Speech Recognition in Spanish)
[Link to Paper - Coming Soon!]

## Introduction
Spanish-AVSR is an adaptation of AV-HuBERT for Audiovisual Speech Recognition (AVSR) specifically in the Spanish language. Based on **AV-HuBERT**, a self-supervised framework, this project aims to bridge the gap in AVSR technology for Spanish, achieving state-of-the-art performance in lip-reading, ASR, and audiovisual speech recognition for Spanish speakers.

## Citation
If you find Spanish-AVSR useful for your research, please cite our upcoming publication (details to be added here soon).


```BibTeX
@article{buitrago2024spanishavsr,
    author = {Pol Buitrago},
    title = {Spanish-AVSR: Audio-Visual Speech Recognition for Spanish},
    journal = {arXiv preprint (coming soon)},
    year = {2024}
}
```

## Pre-trained and Fine-tuned Models

Checkpoints and models adapted for Spanish-AVSR will be made available [here](link-to-checkpoints).

## Installation

To get started with Spanish-AVSR, set up a virtual environment using Conda:

```bash
conda create -n spanish_avsr python=3.8 -y
conda activate spanish_avsr
```

Then, clone the repository:

```bash
git clone https://github.com/Pol-Buitrago/Spanish-AVSR.git
cd Spanish-AVSR
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

Follow the steps in [`data/preparation`](data/preparation) to preprocess:


## Loading a Pretrained Model

```python
import fairseq
ckpt_path = "/path/to/the/checkpoint.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
```

Aquí tienes la sección del README para la parte de **Decoding and Inference** que incluye información sobre el script `infer_AVSR.py` y cómo usarlo para realizar inferencias en lip reading, ASR, y AVSR:


## Decoding and Inference

For lip reading, ASR, or full AVSR, use the provided script located at `avhubert/infer_AVSR.py`. This script allows you to perform inference on audio-visual inputs, leveraging pre-trained models.

### Usage

To use the inference script, follow these steps:

1. **Set up your environment**: Ensure you have all the necessary dependencies installed and have activated your Conda environment.

2. **Prepare your video file**: You need a video file containing the speaker's face (preferably in MP4 format).

3. **Run the inference script**:
   Execute the script with the required arguments. You can choose from the following task types:
   - `AVSR`: Audio-Visual Speech Recognition
   - `ASR`: Automatic Speech Recognition
   - `VSR`: Visual Speech Recognition
   - `ALL`: Run all tasks sequentially

   Example command to run the script:
   ```bash
   python infer_AVSR.py --video_path /path/to/roi.mp4 --audio_path /path/to/clip.wav --task_type AVSR
   ```

### Predict Function

The main function for prediction is defined as follows:
```python
def predict(task_type, video_path=None, audio_path=None, user_dir="", ckpt_path="", suppress_warnings=True):
```

#### Parameters:
- `task_type`: Specifies the type of task to perform (`AVSR`, `ASR`, `VSR`, or `ALL`).
- `video_path`: Path to the video file containing the lip movement.
- `audio_path`: Path to the audio file corresponding to the video.
- `user_dir`: Directory for user-defined modules (optional).
- `ckpt_path`: Path to the model checkpoint (optional).
- `suppress_warnings`: Flag to suppress warnings during execution.

#### Output:
The function returns the predicted text based on the selected task type.

### Notes
- Ensure that the paths to your video and checkpoint is correctly set.
- The video should contain a clear view of the speaker's face for accurate lip reading.
- The audio should be synchronized with the video for best results.


## License
This project follows the AV-HuBERT LICENSE AGREEMENT by Meta Platforms, Inc. For full terms, see [LICENSE](link-to-license-file).

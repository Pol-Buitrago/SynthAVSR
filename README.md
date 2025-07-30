---

# **SynthAVSR (Improving Audiovisual Speech Recognition with Visual Synthetic Data) 🎤🤖**


## **Introduction**  
SynthAVSR is a research framework designed to explore the potential of synthetic audiovisual data for improving AVSR in low-resource languages, specifically Spanish🇪🇸 and Catalan. It implements a full pipeline for generating realistic synthetic lip videos from audio and static images, and fine-tunes AV-HuBERT for evaluating this data across multiple conditions.

---

## **Fine-tuned Models 🧩**  

Checkpoints and models adapted for our project are available in the table below:

| Modality               | MixSpeech<sub>es</sub>           | RealSpeech<sub>es</sub>          | SynthSpeech<sub>es</sub>      | **SynthSpeech<sub>cat</sub>**          |
|------------------------|-------------------|-------------------|------------------------------|-------------------|
| **AudioVisual**        | [Download](https://zenodo.org/records/14679978/files/MixAVSR_AV.pt?download=1)  | [Download](https://zenodo.org/records/14679978/files/RealAVSR_AV.pt?download=1)  | [Download](https://zenodo.org/records/14679978/files/SynthAVSR(GAN)_AV.pt?download=1)             | [Download](https://zenodo.org/records/14680155/files/CAT-AVSR_AV.pt?download=1)  |
| **Audio-Only**         | [Download](https://zenodo.org/records/14679978/files/MixAVSR_A.pt?download=1)  | [Download](https://zenodo.org/records/14679978/files/RealAVSR_A.pt?download=1)  | [Download](https://zenodo.org/records/14679978/files/SynthAVSR(GAN)_A.pt?download=1)             | [Download](https://zenodo.org/records/14680155/files/CAT-AVSR_A.pt?download=1)  |
| **Visual-Only**        | [Download](https://zenodo.org/records/14679978/files/MixAVSR_V.pt?download=1)  | [Download](https://zenodo.org/records/14679978/files/RealAVSR_V.pt?download=1)  | [Download](https://zenodo.org/records/14679978/files/SynthAVSR(GAN)_V.pt?download=1)             | [Download](https://zenodo.org/records/14680155/files/CAT-AVSR_V.pt?download=1)  |

---

### Model Performance (WER) 🎯

#### AVSR Model Results

| Model                           | LIP-RTVE        | CMU-MOSEAS<sub>ES</sub>        | MuAViC<sub>ES</sub>       |
|---------------------------------|-----------------|--------------------------------|---------------------------|
| **MixSpeech<sub>es</sub>**                     | 8.2%            | 14.2%                          | 15.7%                     |
| **RealSpeech<sub>es</sub>**                    | 9.3%            | 15.4%                          | 16.6%                     |
| **SynthSpeech<sub>es</sub>**     | 21.1%           | 35.2%                          | 39.6%                     | 

| Model                 | AVCAT-Benchmark |
|-----------------------|-----------------|
| **SynthSpeech<sub>cat</sub>**          | 25%             |

---


## **Installation ⚙️**  
To get started with SynthAVSR, set up a Conda environment using the `SynthAVSR.yml` file provided:

1. **Create and activate the environment:**
   ```bash
   conda env create -f SynthAVSR.yml
   conda activate synth_avsr
   ```

2. **Clone the repository:**
   ```bash
   git clone https://github.com/Pol-Buitrago/SynthAVSR.git
   cd SynthAVSR
   git submodule init
   git submodule update
   ```

---

## **Data Preparation 📊**  
Follow the steps in [`preparation`](avhubert/preparation/) to pre-process:

- LRS3 and VoxCeleb2 datasets. For any other dataset, follow an analogous procedure.

Follow the steps in [`clustering`](avhubert/clustering/) (for pre-training only) to create:
- `{train, valid}.km` frame-aligned pseudo label files.  
The `label_rate` is the same as the feature frame rate used for clustering, which is 100Hz for MFCC features and 25Hz for AV-HuBERT features by default.

---

## **Training and Fine-tuning AV-HuBERT Models**  

### **Pre-train an AV-HuBERT model**  
To train a model, run the following command, adjusting paths as necessary:
```sh
$ cd avhubert
$ fairseq-hydra-train --config-dir /path/to/conf/ --config-name conf-name \
  task.data=/path/to/data task.label_dir=/path/to/label \
  model.label_rate=100 hydra.run.dir=/path/to/experiment/pretrain/ \
  common.user_dir=`pwd`
```

### **Fine-tune an AV-HuBERT model with Seq2Seq**  
To fine-tune a pre-trained HuBERT model at `/path/to/checkpoint`, run:
```sh
$ cd avhubert
$ fairseq-hydra-train --config-dir /path/to/conf/ --config-name conf-name \
  task.data=/path/to/data task.label_dir=/path/to/label \
  task.tokenizer_bpe_model=/path/to/tokenizer model.w2v_path=/path/to/checkpoint \
  hydra.run.dir=/path/to/experiment/finetune/ common.user_dir=`pwd`
```

### **Decode an AV-HuBERT model**  
To decode a fine-tuned model, run:
```sh
$ cd avhubert
$ python -B infer_s2s.py --config-dir ./conf/ --config-name conf-name \
  dataset.gen_subset=test common_eval.path=/path/to/checkpoint \
  common_eval.results_path=/path/to/experiment/decode/s2s/test \
  override.modalities=['auido,video'] common.user_dir=`pwd`
```
Parameters like `generation.beam` and `generation.lenpen` can be adjusted to fine-tune the decoding process.

---

## License 📜

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
You can freely share, modify, and distribute the code, but it cannot be used for commercial purposes.

See the [full license text](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

---

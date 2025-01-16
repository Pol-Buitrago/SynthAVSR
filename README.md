---

# **SynthAVSR (Audiovisual Speech Recognition with Synthetic Data) 🎤🤖🌍**

[Link to Paper - Coming Soon! 📄]  
[Link to Thesis - Coming Soon! 📚]

## **Introduction**  
SynthAVSR is an advanced framework for Audiovisual Speech Recognition (AVSR) that leverages synthetic data to bridge the gap in AVSR technology. Building upon **AV-HuBERT**, a self-supervised framework, this project aims to push the boundaries of AVSR by focusing on Spanish🇪🇸 and Catalan languages. It uses a novel approach to generate synthetic audiovisual data for training, with the goal of achieving state-of-the-art performance in lip-reading, ASR, and audiovisual speech recognition. 🌟

---

## **Citation**  
If you find SynthAVSR useful for your research, please cite our upcoming publication (details to be added here soon).

```BibTeX
@article{buitrago2024synthavsr,
    author = {Pol Buitrago},
    title = {SynthAVSR: Leveraging Synthetic Data for Advancing Audiovisual Speech Recognition},
    journal = {arXiv preprint (coming soon)},
    year = {2024}
}
```

---

## **Fine-tuned Models 🧩**  

Checkpoints and models adapted for our project are available in the table below:

| Modality               | MixAVSR           | RealAVSR          | SynthAVSR<sub>GAN</sub>      | CAT-AVSR          |
|------------------------|-------------------|-------------------|------------------------------|-------------------|
| **AudioVisual**        | [Download](link)  | [Download](link)  | [Download](link)             | [Download](link)  |
| **Audio-Only**         | [Download](link)  | [Download](link)  | [Download](link)             | [Download](link)  |
| **Visual-Only**        | [Download](link)  | [Download](link)  | [Download](link)             | [Download](link)  |

---

### Model Performance (WER) 🎯

#### AVSR Model Results

| Model                           | LIP-RTVE        | CMU-MOSEAS<sub>ES</sub>        | MuAViC<sub>ES</sub>       |
|---------------------------------|-----------------|--------------------------------|---------------------------|
| **MixAVSR**                     | 8.2%            | 14.2%                          | 15.7%                     |
| **RealAVSR**                    | 9.3%            | 15.4%                          | 16.6%                     |
| **SynthAVSR<sub>GAN</sub>**     | 21.1%           | 35.2%                          | 39.6%                     | 

| Model                 | AVCAT-Benchmark |
|-----------------------|-----------------|
| **CAT-AVSR**          | 25%             |

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

- LRS3 and VoxCeleb2 datasets

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
  override.modalities=['video'] common.user_dir=`pwd`
```
Parameters like `generation.beam` and `generation.lenpen` can be adjusted to fine-tune the decoding process.

---

## License 📜

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
You can freely share, modify, and distribute the code, but it cannot be used for commercial purposes.

See the [full license text](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

---

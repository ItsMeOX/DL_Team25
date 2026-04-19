# 50.039 Theory and Practice of Deep Learning Project - Team 25

## Unsupervised Anomaly Detection Using Toy Car Audio Data

This project investigates unsupervised anomaly detection on multichannel toy car audio using reconstruction-based deep learning models, including DNN-based, CNN-based, LSTM-based, Transformer-based autoencoders, and CNN-based variational autoencoder. The goal is to learn normal sound patterns and detect deviations via reconstruction error.

## Project Structure

    /outputs/
       /models/                  # Trained model checkpoints
           best_CNN_AE.pth
           ...
       /studies/                 # Optuna hyperparameter tuning results
           cnn_study.pkl

    /report/                     # The report for this project
        report.pdf

    /src/                            # Contains all the notebooks for this project
        /helpers/
            helper_audio_data.py      # Audio loading, preprocessing, spectrogram generation
            helper_eval.py            # Evaluation metrics (PR-AUC, scoring, etc.)
            helper_npy_data.py        # .npy dataset handling and loading
        01-Data_Preparation.ipynb    # Data preprocessing pipeline
        02-DNN_AE.ipynb              # DNN Autoencoder training & evaluation
        03-CNN_AE.ipynb              # CNN Autoencoder training & evaluation
        04-LSTM_AE.ipynb             # LSTM Autoencoder training & evaluation
        05-Transformer_AE.ipynb      # Transformer Autoencoder training & evaluation
        06-CNN_VAE.ipynb      # CNN Variational Autoencoder training & evaluation


    requirements.txt

## Data (Not included in repository)

Due to large file sizes, the dataset is **not tracked in GitHub**. The expected structure is:

### Raw audio data (.wav):

    /ToyCar
        /case1
            /AnomalousSound_IND
                1101010001_ToyCar_case1_ab01_IND_ch1_0001.wav
                ...
            /NormalSound_CNT
                1100110001_ToyCar_case1_normal_CNT_ch1_0001.wav
                ....
            /NormalSound_IND
                1100010001_ToyCar_case1_normal_IND_ch1_0001.wav
                ...
        /case2
            ...
        /case3
            ...
        /case4
            ...

### Processed audio data (.npy):

    /ToyCar
        /npy
            /CNT_SEG
                /case[1-4]
                    [sample_id]_seg[seg_idx].npy
                    ...
            /IND
                /case[1-4]
                    /normal
                        [sample_id].npy
                        ...
                    /anomaly
                        ab[anom_id]_[sample_id].npy
                        ...

## Pipeline Overview

### 1. Data Preparation

- Convert `.wav` files into log-mel spectrograms
- Group separate audio files into 4-channel samples
- Remove silent segments via cropping
- Apply min–max normalization
- Save processed data as `.npy` files

### 2. Model Training

- Training uses **normal data only**
- Validation and test use both **normal and abnormal data**
- Models:
  - DNN Autoencoder
  - CNN Autoencoder
  - Transformer Autoencoder
  - LSTM Autoencoder

### 3. Anomaly Detection

- Reconstruction error used as anomaly score
- Supports L1 and L2 scoring
- Evaluated using **Precision-Recall AUC (PR-AUC)**

## Quick Start

### 1) Create and activate a virtual environment

MacOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python3 -m venv .venv
./.venv/Scripts/activate
```

### 2) Install dependencies

Python version used in this project: `3.10.9`.

```bash
pip install -r requirements.txt
```

### 3) Run notebooks

Run in order:

- `/src/01-Data_Preparation.ipynb`
- `/src/02-DNN_AE.ipynb`
- ... other models

### 4) Outputs

- Models: `/outputs/models/`
- Optuna studies: `/outputs/studies`

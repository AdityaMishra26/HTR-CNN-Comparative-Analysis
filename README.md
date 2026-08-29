# HTR-CNN-Comparative-Analysis
# CNN-BiLSTM-CTC Handwritten Text Recognition

## Overview

This repository contains the implementation and experimental analysis of a
CNN-BiLSTM-CTC based Handwritten Text Recognition (HTR) system.

The model processes handwritten word images and predicts the corresponding
text sequence. A Convolutional Neural Network (CNN) extracts visual features,
a Bidirectional Long Short-Term Memory (BiLSTM) network learns sequential
dependencies, and Connectionist Temporal Classification (CTC) is used for
sequence prediction without requiring character-level alignment.

## Model Architecture

Input Image
(1 × 64 × 256)

↓ CNN Feature Extraction

Conv2D (1 → 64) + BatchNorm + ReLU + MaxPool

↓

Conv2D (64 → 128) + BatchNorm + ReLU + MaxPool

↓

Conv2D (128 → 256) + BatchNorm + ReLU + MaxPool

↓

Conv2D (256 → 256) + BatchNorm + ReLU + MaxPool

↓

Feature Projection
(256 × 4 → 256)

↓

2-Layer Bidirectional LSTM
Hidden Size = 256

↓

Linear Character Classifier

↓

CTC Decoding

## Input

- Image format: Grayscale handwritten word images
- Image size: 64 × 256
- Input channels: 1
- Number of characters: 77
- Output classes: 78 including the CTC blank token

## Training Configuration

- Framework: PyTorch
- Hardware: CUDA GPU
- Batch Size: 64
- Planned Epochs: 50
- Actual Training Epochs: 27
- Initial Learning Rate: 0.0005
- Early Stopping Patience: 7 epochs
- Learning Rate Reduction: Factor of 0.5

## Best Training Result

| Metric | Result |
|---|---:|
| Best Epoch | 20 |
| Best Validation Loss | 0.3093 |
| Exact Match Accuracy | 76.98% |
| Character Error Rate (CER) | 8.59% |
| Word Error Rate (WER) | 23.06% |
| Character Accuracy | 91.41% |
| Word Accuracy | 76.94% |
| Character Precision | 93.31% |
| Character Recall | 92.57% |
| Character F1-Score | 92.94% |

## Results

The model achieved its best validation performance at Epoch 20 with a
validation loss of 0.3093.

Evaluation was performed on 3,831 validation samples, with 2,949 exact
correct predictions.

The results indicate strong character-level recognition performance, while
word-level accuracy decreases for longer handwritten words.

## Repository Structure

```text
HTR-CNN-Comparative-Analysis/
├── config.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
├── data/
├── models/
├── notebooks/
├── results/
└── utils/
# ============================================================
# Configuration for CNN-BiLSTM-CTC Handwritten Text Recognition
# ============================================================

import torch

# -----------------------------
# Device Configuration
# -----------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------
# Image Configuration
# -----------------------------
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 1

# -----------------------------
# Model Configuration
# -----------------------------
CNN_FILTERS = [64, 128, 256, 256]

FEATURE_PROJECTION_INPUT = 256 * 4
FEATURE_PROJECTION_OUTPUT = 256

LSTM_INPUT_SIZE = 256
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
BIDIRECTIONAL = True

# -----------------------------
# Training Configuration
# -----------------------------
BATCH_SIZE = 64
NUM_EPOCHS = 50

LEARNING_RATE = 5e-4

EARLY_STOPPING_PATIENCE = 7

# -----------------------------
# Learning Rate Scheduler
# -----------------------------
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3

# -----------------------------
# CTC Configuration
# -----------------------------
CTC_BLANK_INDEX = 0
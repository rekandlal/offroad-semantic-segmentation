import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "segformer"  # "deeplab", "segformer", "unetpp"
                           # TIP: use "segformer" or "unetpp" on CPU — deeplab is very slow on CPU

# Smaller image = massively faster on CPU (256 takes ~8x longer than 128)
IMAGE_SIZE = 128

# Small batch on CPU to avoid memory thrashing
BATCH_SIZE = 2

# Early stopping will quit before this anyway
NUM_EPOCHS = 50

LEARNING_RATE = 1e-4

TRAIN_IMG_DIR = "data/train/images"
TRAIN_MASK_DIR = "data/train/masks"

VAL_IMG_DIR = "data/val/images"
VAL_MASK_DIR = "data/val/masks"

TEST_IMG_DIR = "data/testImages/images"

# On Windows CPU, num_workers > 0 causes multiprocessing overhead — keep at 0
NUM_WORKERS = 0
PIN_MEMORY = False  # Only useful with GPU

SAVE_PATH = "runs/best_model.pth"

# --- Early Stopping Settings ---
EARLY_STOPPING_PATIENCE = 5     # Stop after 5 epochs with no improvement
EARLY_STOPPING_MIN_DELTA = 1e-4 # Minimum val_loss drop to count as improvement

os.makedirs("runs", exist_ok=True)
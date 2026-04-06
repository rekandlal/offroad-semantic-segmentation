import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "deeplab"  # "deeplab", "segformer", "unetpp"

IMAGE_SIZE = 256
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

TRAIN_IMG_DIR = "data/train/images"
TRAIN_MASK_DIR = "data/train/masks"

VAL_IMG_DIR = "data/val/images"
VAL_MASK_DIR = "data/val/masks"

TEST_IMG_DIR = "data/testImages"

NUM_WORKERS = 2
PIN_MEMORY = True

SAVE_PATH = "runs/model.pth"
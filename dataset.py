import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, self.images[idx])
            mask = np.array(Image.open(mask_path).convert("L"))
            mask = (mask > 0).astype("float32")

            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
            mask = torch.tensor(mask).unsqueeze(0)

            return image, mask

        else:
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
            return image
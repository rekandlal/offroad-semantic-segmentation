import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
 
 
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None, image_size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.transform = transform
        self.image_size = image_size  # FIX: resize all images so batching doesn't crash
 
    def __len__(self):
        return len(self.images)
 
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        # FIX: resize to consistent size
        image = Image.open(img_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        image = np.array(image)
 
        if self.mask_dir:
            mask_name = self.images[idx]
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = Image.open(mask_path).convert("L").resize(
                (self.image_size, self.image_size), Image.NEAREST
            )
            mask = np.array(mask)
            mask = (mask > 0).astype("float32")
 
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
            mask = torch.tensor(mask).unsqueeze(0)
 
            # FIX: apply transform if provided
            if self.transform:
                image = self.transform(image)
 
            return image, mask
 
        else:
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
 
            if self.transform:
                image = self.transform(image)
 
            return image
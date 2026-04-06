import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
 
 
class RandomHorizontalFlip:
    """Randomly flip image and mask horizontally with probability p."""
    def __init__(self, p=0.5):
        self.p = p
 
    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask
 
 
class RandomVerticalFlip:
    """Randomly flip image and mask vertically with probability p."""
    def __init__(self, p=0.5):
        self.p = p
 
    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask
 
 
class RandomRotation:
    """Randomly rotate image and mask by up to `degrees`."""
    def __init__(self, degrees=15):
        self.degrees = degrees
 
    def __call__(self, image, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
        return image, mask
 
 
class ColorJitter:
    """Apply color jitter to image only (not mask)."""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2):
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation
        )
 
    def __call__(self, image, mask):
        image = self.jitter(image)
        return image, mask
 
 
class Compose:
    """Chain multiple augmentations together."""
    def __init__(self, transforms):
        self.transforms = transforms
 
    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
 
 
# Ready-to-use augmentation pipeline for training
train_augmentations = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.3),
    RandomRotation(degrees=10),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
])
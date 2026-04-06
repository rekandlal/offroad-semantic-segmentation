import torch
import torch.nn as nn
import torchvision.models as models
 
 
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
 
        # FIX: pretrained=True is deprecated in newer torchvision.
        # Use weights= parameter instead.
        self.model = models.segmentation.deeplabv3_resnet50(
            weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        )
        # Replace final classifier head for binary segmentation
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
 
    def forward(self, x):
        # DeepLabV3 returns an OrderedDict; extract 'out'
        return self.model(x)["out"]
 
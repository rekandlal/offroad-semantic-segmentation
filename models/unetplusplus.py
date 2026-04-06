import torch
import torch.nn as nn
 
 
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x):
        return self.conv(x)
 
 
class UNetPlusPlus(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
 
        self.conv0_0 = DoubleConv(3, 64)
        self.conv1_0 = DoubleConv(64, 128)
        self.conv2_0 = DoubleConv(128, 256)
 
        self.pool = nn.MaxPool2d(2)
 
        self.up1_0 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up2_0 = nn.ConvTranspose2d(256, 128, 2, 2)
 
        self.conv0_1 = DoubleConv(128, 64)   # 64 (skip) + 64 (up) = 128 in
        self.conv1_1 = DoubleConv(256, 128)  # 128 (skip) + 128 (up) = 256 in
 
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
 
    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
 
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_1)], dim=1))
 
        return self.final(x0_1)
 
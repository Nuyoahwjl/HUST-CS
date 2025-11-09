import torch
import torch.nn as nn
import math

class SRCNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True)
        )
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.reconstruction = nn.Conv2d(32, in_channels, kernel_size=5, padding=2)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.map(x)
        return self.reconstruction(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.reconstruction:
                    nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
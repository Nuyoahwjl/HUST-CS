import torch
import torch.nn as nn

class ESPCN(nn.Module):
    def __init__(self, scale=2, in_channels=3):
        super().__init__()
        self.feature_maps = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.Tanh(),
        )
        self.sub_pixel = nn.Sequential(
            nn.Conv2d(32, in_channels * (scale**2), 3, padding=1),
            nn.PixelShuffle(scale)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.feature_maps(x)
        return self.sub_pixel(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

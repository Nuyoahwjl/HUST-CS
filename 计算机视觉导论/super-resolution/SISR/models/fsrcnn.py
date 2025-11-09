import torch
import torch.nn as nn

class FSRCNN(nn.Module):
    def __init__(self, scale=2, in_channels=3, d=56, s=12, m=4):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, d, 5, padding=2),
            nn.PReLU(d)
        )
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, 1),
            nn.PReLU(s)
        )
        self.mapping = nn.Sequential(
            *sum([[nn.Conv2d(s, s, 3, padding=1), nn.PReLU(s)] for _ in range(m)], [])
        )
        self.expand = nn.Sequential(
            nn.Conv2d(s, d, 1),
            nn.PReLU(d)
        )
        self.deconv = nn.ConvTranspose2d(
            d, in_channels, 9, stride=scale, padding=4, output_padding=scale - 1
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.feature(x)
        x = self.shrink(x)
        x = self.mapping(x)
        x = self.expand(x)
        return self.deconv(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)


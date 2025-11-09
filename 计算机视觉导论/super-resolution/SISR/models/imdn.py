import math
import torch
import torch.nn as nn

# ---------------------
# Building Blocks
# ---------------------

def conv3x3(in_ch, out_ch, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias)

class IMDB(nn.Module):
    """
    Information Multi-Distillation Block (Lite)
    distillation_rate=0.25：每层保留一部分通道作为“蒸馏特征”，其余继续深挖。
    结构：C1->split(d,r)->C2->split(d,r)->C3->split(d,r)->C4，最后concat(d1,d2,d3,c4)->1x1融合->残差
    """
    def __init__(self, nf: int, distillation_rate: float = 0.25, act_slope: float = 0.05):
        super().__init__()
        self.nf = nf
        self.dc = max(1, int(nf * distillation_rate))        # distilled channels
        self.rc = nf - self.dc                               # remaining channels

        self.c1 = conv3x3(nf, nf)
        self.c2 = conv3x3(self.rc, nf)
        self.c3 = conv3x3(self.rc, nf)
        self.c4 = conv3x3(self.rc, nf)

        # 1x1 conv fusion after concatenation of [d1,d2,d3,c4]
        self.c5 = nn.Conv2d(self.dc * 3 + nf, nf, kernel_size=1, bias=True)

        self.act = nn.LeakyReLU(negative_slope=act_slope, inplace=True)

    def forward(self, x):
        out_c1 = self.act(self.c1(x))
        d1, r1 = torch.split(out_c1, [self.dc, self.rc], dim=1)

        out_c2 = self.act(self.c2(r1))
        d2, r2 = torch.split(out_c2, [self.dc, self.rc], dim=1)

        out_c3 = self.act(self.c3(r2))
        d3, r3 = torch.split(out_c3, [self.dc, self.rc], dim=1)

        out_c4 = self.act(self.c4(r3))  # no split on the last one

        out = torch.cat([d1, d2, d3, out_c4], dim=1)
        out = self.c5(out)
        return out + x  # local residual

class Upsampler(nn.Module):
    """EDSR风格的上采样尾部，支持 x2/x3/x4"""
    def __init__(self, scale: int, nf: int, out_ch: int, act_slope: float = 0.05):
        super().__init__()
        m = []
        if (scale & (scale - 1)) == 0:  # power of 2 -> x2/x4
            for _ in range(int(math.log2(scale))):
                m += [nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(act_slope, inplace=True)]
        elif scale == 3:
            m += [nn.Conv2d(nf, nf * 9, 3, 1, 1), nn.PixelShuffle(3), nn.LeakyReLU(act_slope, inplace=True)]
        else:
            raise NotImplementedError(f"Unsupported scale: {scale}")
        m += [conv3x3(nf, out_ch)]
        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x)

# ---------------------
# IMDN-Lite (drop-in)
# ---------------------

class IMDN(nn.Module):
    """
    IMDN
    - 头部：3x3 conv
    - 主体：若干 IMDB 块 + 3x3 融合（长残差）
    - 尾部：PixelShuffle 上采样 + 3x3 输出到RGB/Y
    """
    def __init__(
        self,
        in_channels: int = 3,
        scale: int = 2,
        num_features: int = 50,     # IMDN 通常用 50
        num_blocks: int = 6,        # 6~8 块常见；想更快可减到 4
        distillation_rate: float = 0.25,
        act_slope: float = 0.05
    ):
        super().__init__()
        self.scale = scale

        # shallow feature extraction
        self.head = conv3x3(in_channels, num_features)

        # deep feature extraction
        blocks = [IMDB(num_features, distillation_rate, act_slope) for _ in range(num_blocks)]
        self.body = nn.Sequential(
            *blocks,
            conv3x3(num_features, num_features)  # trunk conv for long skip
        )

        self.tail = Upsampler(scale, num_features, in_channels, act_slope)

        self._initialize_weights

    def forward(self, x):
        fea = self.head(x)
        trunk = self.body(fea)
        fea = fea + trunk  # long skip
        out = self.tail(fea)
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

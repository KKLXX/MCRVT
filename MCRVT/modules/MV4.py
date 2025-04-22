import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# MobileNetV4 Block
# ----------------------
class MV4Block(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio=4, use_attention=None):
        super().__init__()
        hidden_dim = int(in_channels * expansion_ratio)
        self.use_attention = use_attention
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = nn.ReLU6()
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                                 padding=1, groups=hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.act2 = nn.ReLU6()
        if self.use_attention:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim // 4, hidden_dim, kernel_size=1),
                nn.Sigmoid()
            )
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.skip_add = nn.quantized.FloatFunctional() if in_channels == out_channels else None

    def forward(self, x):
        identity = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.dw_conv(x)))
        if self.use_attention:
            attn = self.se(x)
            x = x * attn
        x = self.bn3(self.conv3(x))
        if self.skip_add is not None:
            return self.skip_add.add(x, identity)
        return x


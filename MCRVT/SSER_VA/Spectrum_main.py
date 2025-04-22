import torch.nn as nn
import torch.nn.functional as F
from MCRVT.Versatile_attention.VA_layer import Versatile_attention
from MCRVT.modules.FAUnit import FeatureAggregatorUnit
from MCRVT.modules.MV4 import MV4Block


# ----------------------
# Spectrum-based Model with Versatile Attention
# ----------------------
class SpectrumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.fau1 = FeatureAggregatorUnit(32, 32, aggregate_method="add")
        self.mv4_block1 = MV4Block(32, 64, expansion_ratio=2)
        self.fau2 = FeatureAggregatorUnit(64, 64, aggregate_method="concat")
        self.mv4_block2 = MV4Block(64, 128, expansion_ratio=4)
        self.bi_gru = nn.GRU(64, 128, bidirectional=True, batch_first=True)
        self.Versatile_attention = Versatile_attention()
        self.self_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.fau1(x)
        x = self.mv4_block1(x)
        x = self.fau2(x)
        x = x.mean(dim=2)  # Average over frequency bins
        x = x.permute(0, 2, 1)  # [batch, time, features]
        x, _ = self.bi_gru(x)
        x_local = self.Versatile_attention(x)
        x_global, _ = self.self_attention(x, x, x)
        x = x_global + x_local
        return x
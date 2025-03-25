import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Feature Aggregator Unit
# ----------------------

class FeatureAggregatorUnit(nn.Module):
    def __init__(self, in_channels, out_channels, aggregate_method="add"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.aggregate_method = aggregate_method

    def forward(self, x):
        conv_out = self.conv(x)
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)

        if self.aggregate_method == "add":
            return conv_out + avg_out + max_out
        elif self.aggregate_method == "concat":
            return torch.cat([conv_out, avg_out, max_out], dim=1)
        else:
            raise ValueError("Invalid aggregate method")
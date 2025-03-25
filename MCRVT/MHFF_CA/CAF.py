import torch.nn as nn
from MHFF_CA.CRN_WS import DualContrastiveRecon


# ----------------------
# Feature Fusion with CRN
# ----------------------

class FeatureFusionWithCRN_WS(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.crn_ws = DualContrastiveRecon(feature_dim)

    def forward(self, X, Y):
        # X: WavLM features, Y: SSER-VA features
        fused_features, (recon_X, recon_Y) = self.crn_ws(X, Y)
        return fused_features, recon_X, recon_Y

import torch
import torch.nn as nn
import torch.nn.functional as F
from SSER_VA.Spectrum_main import SpectrumModel
from MHFF_CA.CAF import FeatureFusionWithCRN_WS
from extract.wavlm_extractor import WavLMFeatureExtractor
# ----------------------
# Joint Network Framework
# ----------------------

class JointNetwork(nn.Module):
    def __init__(self, num_classes=7, _feature_dim=768):
        super().__init__()
        self.spectrum_model = SpectrumModel()
        self.wavlm_extractor = WavLMFeatureExtractor()
        self.fc_x = nn.Linear(_feature_dim, 256)
        self.fc_y = nn.Linear(256, 256)  # Output dim of SpectrumModel is 256
        self.CAF = FeatureFusionWithCRN_WS(feature_dim=256)
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            ) for _ in range(3)
        ])

    def forward(self, raw_audio, log_mel):
        wavlm_features = self.wavlm_extractor(raw_audio)
        wavlm_features = F.relu(self.fc_x(wavlm_features))
        log_mel = log_mel.unsqueeze(1)
        sser_features = self.spectrum_model(log_mel)
        sser_features = F.relu(self.fc_y(sser_features))
        fused_features, recon_wavlm, recon_sser = self.CAF(wavlm_features, sser_features)
        scale1_logits = self.classifiers[0](wavlm_features.mean(dim=1))
        scale2_logits = self.classifiers[1](sser_features.mean(dim=1))
        scale3_logits = self.classifiers[2](fused_features.mean(dim=1))
        return {
            'logits': [scale1_logits, scale2_logits, scale3_logits],
            'features': (wavlm_features, sser_features),
            'reconstructions': (recon_wavlm, recon_sser)
        }
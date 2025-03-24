import torch
import torch.nn as nn
import torch.nn.functional as F
from dataload import get_train_loader, get_test_loader
from Versatile_attention import VA_layer
from MHFF_CA import CRN_WS
from wavlm_extractor import WavLMFeatureExtractor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import MCRVTLoss

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
        self.Versatile_attention = VA_layer()
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

# ----------------------
# MCRVTLoss Function in modeles
# ----------------------

# ----------------------
# Feature Fusion with CRN
# ----------------------
class FeatureFusionWithCRN_WS(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.crn_ws = CRN_WS(feature_dim)

    def forward(self, X, Y):
        # X: WavLM features, Y: SSER-VA features
        fused_features, (recon_X, recon_Y) = self.crn_ws(X, Y)
        return fused_features, recon_X, recon_Y


# ----------------------
# Joint Network Framework
# ----------------------
class JointNetwork(nn.Module):
    def __init__(self, num_classes=7, _feature_dim=768):
        super().__init__()
        self.spectrum_model = SpectrumModel()
        self.wavlm_extractor = WavLMFeatureExtractor()

        # Feature projection layers
        self.fc_x = nn.Linear(_feature_dim, 256)
        self.fc_y = nn.Linear(256, 256)  # Output dim of SpectrumModel is 256

        # Feature fusion
        self.CAF = FeatureFusionWithCRN_WS(feature_dim=256)

        # Multi-scale classifiers
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            for _ in range(3)  # Three scales
        ])

    def forward(self, raw_audio, log_mel):
        # Extract features from both modalities
        wavlm_features = self.wavlm_extractor(raw_audio)  # [batch, seq_len, 768]
        wavlm_features = F.relu(self.fc_x(wavlm_features))  # [batch, seq_len, 256]

        # Process spectrogram
        log_mel = log_mel.unsqueeze(1)  # Add channel dim
        sser_features = self.spectrum_model(log_mel)  # [batch, seq_len, 256]
        sser_features = F.relu(self.fc_y(sser_features))

        # Feature fusion and reconstruction
        fused_features, recon_wavlm, recon_sser = self.CAF(wavlm_features, sser_features)

        # Multi-scale predictions
        scale1_logits = self.classifiers[0](wavlm_features.mean(dim=1))
        scale2_logits = self.classifiers[1](sser_features.mean(dim=1))
        scale3_logits = self.classifiers[2](fused_features.mean(dim=1))

        return {
            'logits': [scale1_logits, scale2_logits, scale3_logits],
            'features': (wavlm_features, sser_features),
            'reconstructions': (recon_wavlm, recon_sser)
        }

# ----------------------
# Training and Evaluation
# ----------------------
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch_idx, (raw_audio, log_mel, labels) in enumerate(train_loader):
        raw_audio, log_mel, labels = raw_audio.to(device), log_mel.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(raw_audio, log_mel)
        loss_dict = criterion(outputs, labels)
        loss = loss_dict['total_loss']

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for raw_audio, log_mel, labels in test_loader:
            raw_audio, log_mel, labels = raw_audio.to(device), log_mel.to(device), labels.to(device)

            outputs = model(raw_audio, log_mel)
            loss_dict = criterion(outputs, labels)
            loss = loss_dict['total_loss']
            total_loss += loss.item()

            # Use the final scale (fused features) for evaluation
            _, preds = torch.max(outputs['logits'][2], 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    ua = accuracy_score(all_labels, all_preds)  # UA same as Acc (unweighted)
    p = precision_score(all_labels, all_preds, average='macro')
    r = recall_score(all_labels, all_preds, average='macro')
    wf1 = f1_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='macro')

    return {
        'loss': total_loss / len(test_loader),
        'acc': acc,
        'ua': ua,
        'precision': p,
        'recall': r,
        'wf1': wf1,
        'f1': f1
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointNetwork(num_classes=7).to(device)
    train_loader = get_train_loader()
    test_loader = get_test_loader()
    criterion = MCRVTLoss(alpha=1.0, beta=0.1, lambda_=0.5,
                 xi_o=0.01, xi_alpha=0.1, xi_beta=0.05, num_classes=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    for epoch in range(500):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_metrics = test(model, test_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss = {train_loss:.4f}, "
            f"Test Loss = {test_metrics['loss']:.4f}, "
            f"Acc = {test_metrics['acc']:.4f}, "
            f"P = {test_metrics['precision']:.4f}, "
            f"R = {test_metrics['recall']:.4f}, "
            f"UA = {test_metrics['ua']:.4f}, "
            f"WF1 = {test_metrics['wf1']:.4f}, "
            f"F1 = {test_metrics['f1']:.4f}"
        )
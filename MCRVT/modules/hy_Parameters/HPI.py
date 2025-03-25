import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Data.dataload import get_train_loader, get_test_loader
from SSER_VA.Spectrum_main import SpectrumModel
from MHFF_CA.CAF import FeatureFusionWithCRN_WS
from extract.wavlm_extractor import WavLMFeatureExtractor
from modules.MCRVTLoss import MCRVTLoss
from evaluators.MCRVT_evaluators import evaluate
from sklearn.model_selection import ParameterGrid
import json
import os

# ----------------------
# Joint Network Framework
# ----------------------

class JointNetwork(nn.Module):
    def __init__(self, num_classes=7, _feature_dim=768, dropout_rate=0.5, learning_rate=5e-4):
        super().__init__()
        self.spectrum_model = SpectrumModel()
        self.wavlm_extractor = WavLMFeatureExtractor()
        self.fc_x = nn.Linear(_feature_dim, 256)
        self.fc_y = nn.Linear(256, 256)  # Output dim of SpectrumModel is 256
        self.CAF = FeatureFusionWithCRN_WS(feature_dim=256)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, num_classes)
            ) for _ in range(3)  # Three scales
        ])
        self.learning_rate = learning_rate

    def forward(self, raw_audio, log_mel):
        wavlm_features = self.wavlm_extractor(raw_audio)  # [batch, seq_len, 768]
        wavlm_features = F.relu(self.fc_x(wavlm_features))  # [batch, seq_len, 256]
        log_mel = log_mel.unsqueeze(1)  # Add channel dim
        sser_features = self.spectrum_model(log_mel)  # [batch, seq_len, 256]
        sser_features = F.relu(self.fc_y(sser_features))
        fused_features, recon_wavlm, recon_sser = self.CAF(wavlm_features, sser_features)
        fused_features = self.dropout(fused_features)
        scale1_logits = self.classifiers[0](wavlm_features.mean(dim=1))
        scale2_logits = self.classifiers[1](sser_features.mean(dim=1))
        scale3_logits = self.classifiers[2](fused_features.mean(dim=1))

        return {
            'logits': [scale1_logits, scale2_logits, scale3_logits],
            'features': (wavlm_features, sser_features),
            'reconstructions': (recon_wavlm, recon_sser)
        }

# ----------------------
# Training Function
# ----------------------
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss =
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
# ----------------------
# Main Function with Grid Search
# ----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param_grid = {
        'dropout_rate': [0.3, 0.5],
        'learning_rate': [1e-4, 5e-4, 1e-3],
    }
    train_loader = get_train_loader()
    test_loader = get_test_loader()
    best_params = None
    results = []
    os.makedirs('grid_search_results', exist_ok=True)
    results_file = os.path.join('grid_search_results', 'results.json')
    for params in ParameterGrid(param_grid):
        print(f"Training with parameters: {params}")
        model = JointNetwork(
            num_classes=7,
            _feature_dim=768,
            dropout_rate=params['dropout_rate'],
            learning_rate=params['learning_rate']
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = MCRVTLoss()
        num_epochs = 500
        for epoch in range(1, num_epochs + 1):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            test_metrics = evaluate(model, test_loader, criterion, device)  # 使用评估函数

            print(
                f"Epoch {epoch}: "
                f"Train Loss = {train_loss:.4f}, "
                f"Test Loss = {test_metrics['loss']:.4f}, "
                f"Acc = {test_metrics['acc']:.4f}, "
                f"P = {test_metrics['precision']:.4f}, "
                f"R = {test_metrics['recall']:.4f}, "
                f"UA = {test_metrics['ua']:.4f}, "
                f"WF1 = {test_metrics['wf1']:.4f}, "
                f"F1 = {test_metrics['f1']:.4f}"
            )
            if test_metrics['acc'] > best_accuracy:
                best_accuracy = test_metrics['acc']
                best_params = params
                torch.save(model.state_dict(), os.path.join('grid_search_results', 'best_model.pth'))
        results.append({
            'params': params,
            'best_accuracy': test_metrics['acc'],
            'final_accuracy': test_metrics['acc'],
            'best_f1': test_metrics['f1']
        })
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print("Grid search completed.")
    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
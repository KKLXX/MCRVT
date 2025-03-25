import torch
import torch.optim as optim
from MCRVT.Datasets.Datapre.dataload import get_train_loader, get_test_loader
from modules.MCRVTLoss import MCRVTLoss
from evaluators.MCRVT_evaluators import evaluate

#----------------------
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
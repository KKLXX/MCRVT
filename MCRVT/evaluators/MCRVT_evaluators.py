import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (raw_audio, log_mel, labels) in enumerate(test_loader):
            raw_audio, log_mel, labels = raw_audio.to(device), log_mel.to(device), labels.to(device)
            outputs = model(raw_audio, log_mel)
            loss_dict = criterion(outputs, labels)
            loss = loss_dict['total_loss']
            total_loss += loss.item()
            _, preds = torch.max(outputs['logits'][2], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    ua = accuracy_score(all_labels, all_preds)
    p = precision_score(all_labels, all_preds, average='macro')
    r = recall_score(all_labels, all_preds, average='macro')
    wf1 = f1_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='macro')
    metrics = {
        'loss': total_loss / len(test_loader),
        'acc': acc,
        'ua': ua,
        'precision': p,
        'recall': r,
        'wf1': wf1,
        'f1': f1
    }
    return metrics
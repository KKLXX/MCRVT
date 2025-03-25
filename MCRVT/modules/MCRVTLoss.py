import torch
import torch.nn as nn
import torch.nn.functional as F

class MCRVTLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, lambda_=0.5,
                 xi_o=0.01, xi_alpha=0.1, xi_beta=0.05,
                 num_scales=3, num_classes=6):
        """
        Multi-scale loss function for MCRVT

        Parameters (default values from paper):
            alpha: weight for multi-scale loss (1.0)
            beta: weight for contrastive loss (0.1)
            lambda_: weight for reconstruction loss (0.5)
            xi_o: offset parameter (0.01)
            xi_alpha: alpha parameter (0.1)
            xi_beta: beta parameter (0.05)
            num_scales: number of scales (3)
            num_classes: number of emotion classes
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_
        self.xi_o = xi_o
        self.xi_alpha = xi_alpha
        self.xi_beta = xi_beta
        self.num_scales = num_scales
        self.num_classes = num_classes


        self.focal_losses = nn.ModuleList([
            FocalLoss(gamma=2, reduction='mean')
            for _ in range(num_scales)
        ])


        self.contrastive_loss = SupConLoss()

    def forward(self, outputs, targets):
        """
        Compute the final MCRVT loss

        Args:
            outputs: Dictionary containing:
                - logits: List of logits from each scale [scale1, scale2, scale3]
                - features: Tuple of (WavLM features, SSER-VA features)
                - reconstructions: Tuple of (reconstructed_WavLM, reconstructed_SSERVA)
            targets: Ground truth emotion labels
        """
        if len(outputs['logits']) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} scales, got {len(outputs['logits'])}")
        L_ms = 0
        scale_weights = torch.arange(1, self.num_scales + 1, dtype=torch.float32)
        scale_weights = scale_weights / scale_weights.sum()  # Normalize

        for i, (logits, floss) in enumerate(zip(outputs['logits'], self.focal_losses)):
            L_ms += scale_weights[i] * floss(logits, targets)
        features_w, features_s = outputs['features']
        L_cl = self.contrastive_loss(
            torch.cat([features_w.unsqueeze(1), features_s.unsqueeze(1)], dim=1),
            targets
        recon_w, recon_s = outputs['reconstructions']
        L_RI = F.mse_loss(recon_w, features_w) + F.mse_loss(recon_s, features_s)

        total_loss = (self.alpha * L_ms +
                      self.beta * L_cl +
                      self.lambda_ * L_RI)

        return {
            'total_loss': total_loss,
            'L_ms': L_ms.detach(),
            'L_cl': L_cl.detach(),
            'L_RI': L_RI.detach()
        }


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        bsz = features.shape[0]

        features = F.normalize(features, p=2, dim=2)
        features = features.view(bsz, -1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature


        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()


        self_mask = torch.eye(bsz, device=features.device)
        mask = mask * (1 - self_mask)

        exp_sim = torch.exp(sim_matrix) * (1 - self_mask)  # subtract self
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))


        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos.mean()
        return loss

def export_loss_function(config=None):
    """
    Export the loss function with optional configuration

    Args:
        config: Dictionary with optional parameters:
            - alpha: weight for multi-scale loss
            - beta: weight for contrastive loss
            - lambda_: weight for reconstruction loss
            - xi_o: offset parameter
            - xi_alpha: alpha parameter
            - xi_beta: beta parameter
            - num_scales: number of scales
            - num_classes: number of classes

    Returns:
        Initialized MCRVTLoss instance
    """
    default_config = {
        'alpha': 1.0,
        'beta': 0.1,
        'lambda_': 0.5,
        'xi_o': 0.01,
        'xi_alpha': 0.1,
        'xi_beta': 0.05,
        'num_scales': 3,
        'num_classes': 6
    }

    if config:
        default_config.update(config)

    return MCRVTLoss(**default_config)
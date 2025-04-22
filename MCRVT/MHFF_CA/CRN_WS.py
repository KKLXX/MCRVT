import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveReconstructionNetwork(nn.Module):
    """
    Contrastive Reconstruction Networks (CRN) for cross-modal feature reconstruction
    
    Args:
        input_dim_w (int): Dimension of WavLM features
        input_dim_s (int): Dimension of SSER-VA features
        latent_dim (int): Dimension of latent space
        num_labels (int): Number of emotion classes
        temperature (float): Temperature parameter for contrastive loss
    """
    def __init__(self, input_dim_w, input_dim_s, latent_dim=256, num_labels=6, temperature=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.temperature = temperature
        
        # Modality-specific encoders (2-layer MLPs)
        self.encoder_w = nn.Sequential(
            nn.Linear(input_dim_w, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        self.encoder_s = nn.Sequential(
            nn.Linear(input_dim_s, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Modality-specific decoders (2-layer MLPs)
        self.decoder_w = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, input_dim_w)
        )
        
        self.decoder_s = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, input_dim_s)
        )
        
        # Bi-reconstruction networks
        self.g_ws = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        self.g_sw = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        # Intrinsic vectors for each label
        self.d_w = nn.Parameter(torch.randn(num_labels, latent_dim))
        self.d_s = nn.Parameter(torch.randn(num_labels, latent_dim))
        
        # Classifiers
        self.classifier_w = nn.Linear(latent_dim, num_labels)
        self.classifier_s = nn.Linear(latent_dim, num_labels)
        
    def forward(self, x_w, x_s, labels=None, queue=None):
        """
        Forward pass of CRN
        
        Args:
            x_w (Tensor): WavLM features [batch_size, seq_len, input_dim_w]
            x_s (Tensor): SSER-VA features [batch_size, seq_len, input_dim_s]
            labels (Tensor): Emotion labels [batch_size]
            queue (Tensor): Queue for contrastive learning [queue_size, latent_dim]
            
        Returns:
            dict: Contains reconstructed features and losses
        """
        # Encode features to latent space
        z_w = self.encoder_w(x_w)  # [batch_size, seq_len, latent_dim]
        z_s = self.encoder_s(x_s)  # [batch_size, seq_len, latent_dim]
        
        # Get intrinsic vectors for current batch
        if labels is not None:
            d_w = self.d_w[labels]  # [batch_size, latent_dim]
            d_s = self.d_s[labels]  # [batch_size, latent_dim]
        else:
            d_w = self.d_w.mean(0).expand(x_w.size(0), -1)  # [batch_size, latent_dim]
            d_s = self.d_s.mean(0).expand(x_s.size(0), -1)  # [batch_size, latent_dim]
        
        # Decode back to feature space
        x_w_recon = self.decoder_w(z_w)  # [batch_size, seq_len, input_dim_w]
        x_s_recon = self.decoder_s(z_s)  # [batch_size, seq_len, input_dim_s]
        d_w_recon = self.decoder_w(d_w)  # [batch_size, latent_dim]
        d_s_recon = self.decoder_s(d_s)  # [batch_size, latent_dim]
        
        # Cross-modal reconstruction
        # Combine WavLM distribution info with SSER-VA features
        c_ws = torch.cat([d_w_recon.unsqueeze(1).expand(-1, x_s.size(1), -1), 
                         x_s_recon], dim=-1)  # [batch_size, seq_len, latent_dim + input_dim_s]
        
        # Combine SSER-VA distribution info with WavLM features
        c_sw = torch.cat([d_s_recon.unsqueeze(1).expand(-1, x_w.size(1), -1), 
                         x_w_recon], dim=-1)  # [batch_size, seq_len, latent_dim + input_dim_w]
        
        # First-stage reconstruction
        x_alpha_w = self.g_ws(c_ws)  # [batch_size, seq_len, latent_dim]
        x_alpha_s = self.g_sw(c_sw)  # [batch_size, seq_len, latent_dim]
        
        # Second-stage reconstruction
        x_beta_w = self.g_ws(torch.cat([x_alpha_w, x_alpha_s], dim=-1))  # [batch_size, seq_len, latent_dim]
        x_beta_s = self.g_sw(torch.cat([x_alpha_s, x_alpha_w], dim=-1))  # [batch_size, seq_len, latent_dim]
        
        # Classification results
        logits_w = self.classifier_w(z_w.mean(1))  # [batch_size, num_labels]
        logits_s = self.classifier_s(z_s.mean(1))  # [batch_size, num_labels]
        
        # Prepare outputs
        outputs = {
            'x_alpha_w': x_alpha_w,
            'x_alpha_s': x_alpha_s,
            'x_beta_w': x_beta_w,
            'x_beta_s': x_beta_s,
            'logits_w': logits_w,
            'logits_s': logits_s
        }
        
        # Calculate losses if labels are provided
        if labels is not None:
            # Contrastive loss
            contrastive_loss = self.compute_contrastive_loss(z_w, z_s, x_alpha_w, x_alpha_s, 
                                                           x_beta_w, x_beta_s, queue)
            
            # Reconstruction loss (binary cross entropy)
            recon_loss = self.compute_reconstruction_loss(logits_w, logits_s, labels)
            
            outputs.update({
                'contrastive_loss': contrastive_loss,
                'recon_loss': recon_loss
            })
        
        return outputs
    
    def compute_contrastive_loss(self, z_w, z_s, x_alpha_w, x_alpha_s, x_beta_w, x_beta_s, queue=None):
        """
        Compute contrastive loss using all feature variants
        
        Args:
            z_w, z_s: Original encoded features
            x_alpha_w, x_alpha_s: First-stage reconstructed features
            x_beta_w, x_beta_s: Second-stage reconstructed features
            queue: Queue of past embeddings for contrastive learning
            
        Returns:
            Tensor: Contrastive loss value
        """
        # Flatten all features to [batch_size * seq_len, latent_dim]
        all_features = []
        for feat in [z_w, z_s, x_alpha_w, x_alpha_s, x_beta_w, x_beta_s]:
            all_features.append(feat.reshape(-1, self.latent_dim))
        
        # Combine with queue if provided
        if queue is not None:
            all_features.append(queue)
        
        # Stack all features
        contrastive_pool = torch.cat(all_features, dim=0)  # [N, latent_dim]
        
        # Normalize features
        contrastive_pool = F.normalize(contrastive_pool, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(contrastive_pool, contrastive_pool.T) / self.temperature
        
        # For each sample, positive samples are other views of same sample
        batch_size = z_w.size(0) * z_w.size(1)
        labels = torch.arange(batch_size, device=z_w.device)
        labels = labels.repeat(6)  # 6 variants per sample
        
        # Mask for positive samples
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        mask = mask - torch.eye(mask.size(0), device=mask.device)
        
        # Log-softmax
        log_probs = F.log_softmax(sim_matrix, dim=-1)
        
        # Compute contrastive loss
        loss = - (mask * log_probs).sum(1) / mask.sum(1)
        loss = loss.mean()
        
        return loss
    
    def compute_reconstruction_loss(self, logits_w, logits_s, labels):
        """
        Compute reconstruction loss using BCE on classifier outputs
        
        Args:
            logits_w, logits_s: Classifier outputs for each modality
            labels: Ground truth labels
            
        Returns:
            Tensor: Reconstruction loss value
        """
        # Convert labels to one-hot
        targets = F.one_hot(labels, num_classes=self.d_w.size(0)).float()
        
        # BCE losses for each stage
        loss_w = F.binary_cross_entropy_with_logits(logits_w, targets)
        loss_s = F.binary_cross_entropy_with_logits(logits_s, targets)
        
        # Weighted sum (weights from paper: 0.01, 0.1, 0.05)
        total_loss = 0.01 * loss_w + 0.1 * loss_s
        
        return total_loss

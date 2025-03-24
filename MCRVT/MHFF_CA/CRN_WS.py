import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TransformerEncoder, TransformerEncoderLayer
from Cross_Attention import CA

class LabelWiseAttention(nn.Module):
    def __init__(self, d_model, n_labels):
        super().__init__()
        self.attention = nn.Linear(d_model, n_labels)
    def forward(self, h):
        # h: [seq_len, batch, d_model]
        attn_weights = F.softmax(self.attention(h), dim=1)  # [seq_len, batch, n_labels]
        u = torch.einsum('sbd,sbn->bnd', h, attn_weights)  # [batch, n_labels, d_model]
        return u
class DualModalityEncoder(nn.Module):
    def __init__(self, input_dims, d_model, n_labels, nhead=4):
        super().__init__()
        # W2
        self.encoder_w = nn.Sequential(
            nn.Linear(input_dims['w'], d_model),
            TransformerEncoder(
                TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4),
                num_layers=6
            )
        )
        # Sp
        self.encoder_s = nn.Sequential(
            nn.Linear(input_dims['s'], d_model),
            TransformerEncoder(
                TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4),
                num_layers=4
            )
        )
        self.attention = LabelWiseAttention(d_model, n_labels)
    def forward(self, x_w, x_s):
        # x_w: [seq_len, batch, dim_w]
        # x_s: [seq_len, batch, dim_s]
        h_w = self.encoder_w(x_w)  # [seq_len, batch, d_model]
        h_s = self.encoder_s(x_s)
        u_w = self.attention(h_w)  # [batch, n_labels, d_model]
        u_s = self.attention(h_s)
        return {'w': u_w, 's': u_s}
class DualContrastiveRecon(nn.Module):
    def __init__(self, d_model, d_latent, n_labels):
        super().__init__()
        # 编码器/解码器
        self.encoders = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(d_model, d_latent),
                nn.ReLU(),
                nn.Linear(d_latent, d_latent)
            ) for m in ['w', 's']
        })
        self.decoders = nn.ModuleDict({
            m: nn.Linear(d_latent, d_model) for m in ['w', 's']
        })
        self.recon_net = nn.ModuleDict({
            's2w': nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU()),
            'w2s': nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU())
        })
        self.queue_size = 8192
        self.register_buffer("queue", torch.randn(2 * d_latent, self.queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    def forward(self, u_dict):
        z_dict = {m: self.encoders[m](u_dict[m]) for m in ['w', 's']}
        u_tilde_dict = {m: self.decoders[m](z_dict[m]) for m in ['w', 's']}
        u_alpha_w = self.recon_net['s2w'](torch.cat([u_tilde_dict['w'], u_dict['s']], dim=-1))
        u_alpha_s = self.recon_net['w2s'](torch.cat([u_dict['w'], u_tilde_dict['s']], dim=-1))
        self._update_queue(torch.cat([z_dict['w'], z_dict['s']], dim=-1))
        return {
            'u_origin': u_dict,
            'u_alpha': {'w': u_alpha_w, 's': u_alpha_s},
            'z_dict': z_dict
        }
    def _update_queue(self, keys):
        keys = keys.detach()
        ptr = int(self.queue_ptr)
        batch_size = keys.shape[0]
        if ptr + batch_size > self.queue_size:
            self.queue[:, ptr:] = keys[:self.queue_size - ptr].transpose(0, 1)
            self.queue[:, :(ptr + batch_size - self.queue_size)] = keys[self.queue_size - ptr:].transpose(0, 1)
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
class CRN_Dual(nn.Module):
    def __init__(self, input_dims, d_model=256, d_latent=64, n_labels=6):
        super().__init__()
        self.encoder = DualModalityEncoder(input_dims, d_model, n_labels)
        self.reconstructor = DualContrastiveRecon(d_model, d_latent)
        self.cross_attn = CA(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, n_labels),
            nn.Sigmoid()
        )
        self.modal_cls = nn.ModuleDict({
            'w': nn.Linear(d_model, n_labels),
            's': nn.Linear(d_model, n_labels)
        })
    def forward(self, x_w, x_s, labels=None):
        u_dict = self.encoder(x_w, x_s)  # {'w': [B,N,D], 's': [B,N,D]}
        recon = self.reconstructor(u_dict)
        fused = self.cross_attn(
            recon['u_alpha']['w'],
            recon['u_alpha']['s']
        )
        return fused

import torch
import torch.nn as nn
import torch.nn.functional as F

class CA2(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.W_xq = nn.Linear(feature_dim, feature_dim)
        self.W_yq = nn.Linear(feature_dim, feature_dim)
        self.W_fk = nn.Linear(feature_dim * 2, feature_dim)
        self.W_fv = nn.Linear(feature_dim * 2, feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, X, Y):
        Q1 = self.W_xq(X)
        Q2 = self.W_yq(Y)
        Q_cat = torch.cat([Q1, Q2], dim=-1)
        K_f = self.W_fk(Q_cat)
        V_f = self.W_fv(Q_cat)
        U_A1 = X + self.dropout(
            F.softmax(torch.bmm(Q1, K_f.transpose(1, 2)) / (self.feature_dim ** 0.5), dim=-1) @ V_f
        )
        U_A2 = Y + self.dropout(
            F.softmax(torch.bmm(Q2, K_f.transpose(1, 2)) / (self.feature_dim ** 0.5), dim=-1) @ V_f
        )
        X_out = self.dropout(U_A1 + U_A2)
        return X_out
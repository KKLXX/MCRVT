import torch
import torch.nn as nn
import torch.nn.functional as F

class CA(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.W_xq = nn.Linear(feature_dim, feature_dim)
        self.W_xk = nn.Linear(feature_dim, feature_dim)
        self.W_xv = nn.Linear(feature_dim, feature_dim)
        self.W_yq = nn.Linear(feature_dim, feature_dim)
        self.W_yk = nn.Linear(feature_dim, feature_dim)
        self.W_yv = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, X, Y):
        Q1, K1, V1 = self.W_xq(X), self.W_xk(X), self.W_xv(X)
        Q2, K2, V2 = self.W_yq(Y), self.W_yk(Y), self.W_yv(Y)
        U_A1 = X + self.dropout(F.softmax(torch.bmm(Q1, K2.transpose(1, 2)) / (self.feature_dim ** 0.5)) @ V2)
        U_A2 = Y + self.dropout(F.softmax(torch.bmm(Q2, K1.transpose(1, 2)) / (self.feature_dim ** 0.5)) @ V1)
        X_out = self.dropout(U_A1 + U_A2)
        return X_out
import torch.nn as nn
from configs.config import DROPOUT

class SublayerConnection(nn.Module):
    """残差包装器：out = layer_norm(x + sublayer(x))"""
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


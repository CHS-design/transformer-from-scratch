import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """位置编码：sin/cos，与词向量拼接。"""
    def __init__(self, d_model, max_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len,d_model)
        postition = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )

        pe[:, 0::2] = torch.sin(postition * div_term)
        pe[:, 1::2] = torch.cos(postition * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)


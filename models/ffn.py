import torch.nn as nn
from configs.config import DROPOUT

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=DROPOUT):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_2(self.relu(self.w_1(x))))

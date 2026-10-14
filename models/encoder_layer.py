import torch.nn as nn
from models.attention import MultiHeadAttention
from models.ffn import PositionwiseFeedForward
from models.sublayer_connection import SublayerConnection

class EncoderLayer(nn.Module):
    """一层 Encoder = 自注意力子层 + FFN 子层，各配一个残差包装器。"""
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayers= nn.ModuleList(
            [SublayerConnection(d_model, dropout) for _ in range(2)]
        )

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask)[0])
        x = self.sublayers[1](x, self.feed_forward)
        return x
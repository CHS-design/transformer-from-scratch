import torch.nn as nn
from models.attention import MultiHeadAttention
from models.ffn import PositionwiseFeedForward
from models.sublayer_connection import SublayerConnection

class DecoderLayer(nn.Module):
    """一层 DecoderLayer = 自注意力子层 + 交叉注意力层 + FFN 子层，各配一个残差包装器。"""
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayers= nn.ModuleList(
            [SublayerConnection(d_model, dropout) for _ in range(3)]
        )

    def forward(self, x, trg_mask, src_mask, memory):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, trg_mask)[0])
        x = self.sublayers[1](x, lambda x: self.cross_attn(x, memory, memory, src_mask)[0])
        x = self.sublayers[2](x, self.feed_forward)
        return x
import math
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """打分 → 缩放 → 盖章 → 预算分配 → 加权求和。零参数，纯数学。"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, Q, K, V, mask = None):

        scores = torch.matmul(Q, K.transpose(-1, -2))
        scores = scores / math.sqrt(Q.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(scores, dim = -1)

        return torch.matmul(attention, V), attention

class MultiHeadAttention(nn.Module):
    """多头注意力机制。"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def _split_heads(self,x):
            """(batch_size, seq_len, d_model) → (batch_size, n_heads, seq_len, d_k)"""
            batch_size = x.size(0)
            x = x.view(batch_size, -1, self.n_heads, self.d_k)
            return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
            """(batch_size, n_heads, seq_len, d_k) → (batch_size, seq_len, d_model)"""
            batch_size = x.size(0)
            x = x.permute(0, 2, 1, 3).contiguous()
            return x.view(batch_size, -1, self.n_heads * self.d_k)

    def forward(self, Q, K, V, mask = None):
            Q = self._split_heads(self.W_Q(Q))
            K = self._split_heads(self.W_K(K))
            V = self._split_heads(self.W_V(V))

            attn_output, attention = self.attention(Q, K, V, mask)

            return self.W_O(self._merge_heads(attn_output)), attention

        
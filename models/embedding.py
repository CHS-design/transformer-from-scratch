import math
import torch.nn as nn
from utils.vocab import PAD_IDX

class TokenEmbedding(nn.Module):
   """id → 语义向量：查一张可学习的 (vocab_size, d_model) 大表，再放大 √d_model。"""
   def __init__(self, vocab_size, d_model):
      super().__init__()
      self.embed = nn.Embedding(vocab_size, d_model, padding_idx= PAD_IDX)
      self.d_model = d_model

   def forward(self, tokens):
      return self.embed(tokens) * math.sqrt(self.d_model)




import torch.nn as nn
from models.embedding import TokenEmbedding
from models.positional_encoding import PositionalEncoding
from models.encoder import Encoder
from models.decoder import Decoder
from models.decoder_layer import DecoderLayer
from models.encoder_layer import EncoderLayer

class Transformer(nn.Module):
    """Transformer模型:双词表Embedding + 共享PE + N层Encoder + N层Decoder + 1层Linear"""
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, n_heads, n_layers,d_ff, dropout, max_len):
        super().__init__()
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.trg_embedding = TokenEmbedding(trg_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, n_heads, d_ff, dropout), n_layers)
        self.decoder = Decoder(DecoderLayer(d_model, n_heads, d_ff, dropout), n_layers)
        self.output = nn.Linear(d_model, trg_vocab_size)

    def encode(self, src, src_mask):
        x = self.src_embedding(src)
        x = self.pos_encoder(x)
        return self.encoder(x, src_mask)

    def decode(self, trg, trg_mask, src_mask, memory):
        x = self.trg_embedding(trg)
        x = self.pos_encoder(x)
        return self.decoder(x, trg_mask, src_mask, memory)

    def forward(self, src, trg, src_mask, trg_mask):
        memory = self.encode(src, src_mask)
        return self.output(self.decode(trg, trg_mask, src_mask, memory))


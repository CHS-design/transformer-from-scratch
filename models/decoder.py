import copy
import torch.nn as nn

class Decoder(nn.Module):
    """N层decoderlayer堆叠而成"""

    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

    def forward(self, x, trg_mask, src_mask, memory):
        for layer in self.layers:
            x = layer(x, trg_mask, src_mask, memory)
        return x
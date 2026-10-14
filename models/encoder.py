import copy
import torch.nn as nn

class Encoder(nn.Module):
    """N层Encoderlayer堆叠而成"""

    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x
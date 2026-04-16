import torch.nn as nn
from .encoder_layer import EncoderLayer

class TransformerEncoder(nn.Module):
    def __init__(self, d_model = 512, num_head = 8, num_layer = 6, drop_out = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_head, drop_out) for _ in range(num_layer)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
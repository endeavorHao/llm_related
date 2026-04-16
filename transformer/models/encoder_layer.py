import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_head, drop_out: float= 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_head, drop_out)
        self.ff = FeedForward(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ff(x))
        return x
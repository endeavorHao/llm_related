import torch
from models.transformer import TransformerEncoder
import Config

model = TransformerEncoder(
    d_model = Config.d_model,
    num_head = Config.num_head,
    num_layer = Config.num_layer,
    drop_out = Config.drop_out
)

x = torch.randn(2, 10, Config.d_model)
output = model(x)
print(output.shape)
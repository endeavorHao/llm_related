import torch 
import torch.nn as nn
import torch.nn.functional as F 
from dataclasses import dataclass

@dataclass
class DataConfig:
    batch_size: int = 32
    seq_len: int = 128
    in_features: int = 768
    out_features: int = 512
    rank: int = 8
    lora_alpha: int = 16
    dropout: float = 0.1
    merge: bool = True

class LinearLoRALayer(nn.Module):
    def __init__(self, Config):
        super().__init__()

        self.in_features = Config.in_features
        self.out_features = Config.out_features
        self.rank = Config.rank
        self.merge = Config.merge
        self.lora_alpha = Config.lora_alpha
        self.dropout = nn.Dropout(Config.dropout) if Config.dropout > 0 else nn.Identity()
    

        # linear weight shape is (out_features, in_features) 是 x W^T
        self.ln = nn.Linear(Config.in_features, Config.out_features)

        if Config.rank > 0:
            # 这里是为了标记lora_a 和 lora_b 是可训练的参数
            # lora_a shape is (out_features, rank), lora_b shape is (rank, in_features)
            self.lora_a = nn.Parameter(
                torch.zeros(Config.out_features, Config.rank)
            )
            # lora_a 需要初始化为 高斯分布
            nn.init.kaiming_normal_(self.lora_a, a=0.01)

            self.lora_b = nn.Parameter(
                torch.zeros(Config.rank, Config.in_features)
            )
            self.scale = Config.lora_alpha / Config.rank

            # linear 需要设置成不可以训练
            self.ln.weight.requires_grad = False
            self.ln.bias.requires_grad = False

        if Config.merge:
            self.merge_weight()


    def forward(self, x):
        # x shape is (batch_size, seq_len, in_features)
        # in_features 相当于是 embdding_size
        if self.rank > 0 and not self.merge:
            output = self.ln(x) + self.scale * (x @ (self.lora_a @ self.lora_b).T)
        elif self.rank > 0 and self.merge:
            output = self.ln(x)
        else:   
            output = self.ln(x)
        
        return self.dropout(output)
    
    def merge_weight(self, ):
        if self.merge and self.rank > 0:
            self.ln.weight.data += self.scale * (self.lora_a @ self.lora_b)
    
    def unmerge_weight(self, ):
        if self.rank > 0:
            self.ln.weight.data -= self.scale * (self.lora_a @ self.lora_b)

x = torch.randn(DataConfig.batch_size, DataConfig.seq_len, DataConfig.in_features)
lora_layer = LinearLoRALayer(DataConfig)

output = lora_layer(x)
print(output.shape)
# output shape is (batch_size, seq_len, out_features)

lora_layer.merge_weight
output_after_merge = lora_layer(x)
lora_layer.unmerge_weight
output_after_unmerge = lora_layer(x)

print("Max difference after merge/unmerge cycle:", 
      torch.max(torch.abs(output - output_after_unmerge)).item())
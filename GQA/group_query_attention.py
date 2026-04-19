import torch
import torch.nn as nn
import math
from dataclasses import dataclass

@dataclass
class DataConfig:
    batch_size: int = 3
    seq_len: int = 2
    hidden_dim: int = 128
    nums_head: int = 8
    nums_key_value_head: int = 4


class GroupQueryAttention(nn.Module):
    def __init__(self, Config):
        super().__init__()
        # nums_head 表示q的头， nums_key_value_head 表示N个query头为一组
        assert Config.hidden_dim % Config.nums_head == 0
        assert Config.nums_head % Config.nums_key_value_head == 0

        self.hidden_size = Config.hidden_dim
        self.nums_head = Config.nums_head
        self.nums_key_value_head = Config.nums_key_value_head
        self.head_dim = Config.hidden_dim // Config.nums_head

        self.q = nn.Linear(Config.hidden_dim, Config.hidden_dim)
        self.k = nn.Linear(Config.hidden_dim, self.head_dim * self.nums_key_value_head)
        self.v = nn.Linear(Config.hidden_dim, self.head_dim * self.nums_key_value_head)
        self.o = nn.Linear(Config.hidden_dim, Config.hidden_dim)
        
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()

        # q shape is (batch_size, seq_len, hidden_dim)
        # kv shape is (batch_size, seq_len, head_dim * nums_key_value_head)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nums_key_value_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nums_key_value_head, self.head_dim).transpose(1, 2)

        # repeat 广播操作
        k = k.repeat_interleave(self.nums_head // self.nums_key_value_head, dim = 1)
        v = v.repeat_interleave(self.nums_head // self.nums_key_value_head, dim = 1)
        
        attention_score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 忽略attention_mask
        attention_weight = torch.softmax(attention_score, dim=-1)
        # 忽略dropout

        output = attention_weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, hidden_dim)
        output = self.o(output)

        return output

x = torch.rand(3, 2, 128)
net = GroupQueryAttention(DataConfig)
print(net(x).shape)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head):
        super().__init__()
        assert hidden_dim % nums_head == 0

        # nums_head 表示一个attention有多少个头
        # hidden_dim 就是一个token分为向量长度
        # head_dim 表示一个attention表示有多少的向量维度
        self.nums_head = nums_head
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // nums_head

        self.query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)

        # gpt2 和 bert 都有, 但是llama 没有
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x, attention_mask=None):
        # X shape is (batch_size, seq_len, hidden_dim)
        # attention_mask shape is (seq_len, seq_len)

        batch_size, seq_len, hidden_dim = x.size()

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # 需要将 qkv shape 从(batch_size, seq_len, hidden_dim) -> (batch_size, nums_head, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)

        attention_weight = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        print(type(attention_weight))
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(attention_mask == 0, float("-1e9")) 

        attention_weight = F.softmax(attention_weight, dim=-1)
        print(attention_weight)
        attention_weight = self.dropout(attention_weight)
        output = attention_weight @ v
        
        # output shape is (batch_size, nums_head, seq_len, head_dim) -> (batch_size, seq_len, hidden_dim)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, hidden_dim)
        output = self.out(output)
        return output


batch, seq, dim, heads = 3, 2, 128, 8
# 构造一个 (B, 1, S, S) 的 mask
# 1 表示关注，0 表示遮盖
mask = torch.ones(batch, 1, seq, seq)
mask[:, :, 1, 0] = 0 # 举例：让第2个词看不见第1个词

x = torch.rand(batch, seq, dim)
net = MultiHeadSelfAttention(dim, heads)
out = net(x, mask)
print(f"输出形状: {out.shape}") # 预期: torch.Size([3, 2, 128])
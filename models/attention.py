import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head, drop_out = 0.1):
        super().__init__()
        assert(d_model % num_head == 0)
        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_model // num_head

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
    
        self.drop_out = nn.Dropout(drop_out)
        
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, X, mask = None):
        # X: (batch_size, seq_len, d_model)
        b, s, _ = X.size()

        q = self.q_proj(X)
        k = self.k_proj(X)
        v = self.v_proj(X)

        # 将q，k，v (batch_size, seq_len, d_model) 转换成 (batch_size, num_head, seq_len, d_k)
        q = q.view(b, s, self.num_head, self.d_k).transpose(1, 2)
        k = k.view(b, s, self.num_head, self.d_k).transpose(1, 2)
        v = v.view(b, s, self.num_head, self.d_k).transpose(1, 2)

        # attention_weight: (batch_size, num_head, seq_len, seq_len)
        attention_weight = q @ k.transpose(-1, -2) / math.sqrt(self.d_k)
        if mask is not None:    
            attention_weight = attention_weight.masked_fill(mask == 0, -1e9)
        
        attention_weight = torch.softmax(attention_weight, dim = -1)

        # print(attention_weight.shape)
        # print(attention_weight)
        attention_weight = self.drop_out(attention_weight)
        output = attention_weight @ v
        output = output.transpose(1, 2).contiguous()

        output = output.view(b, s, self.d_model)

        output = self.out_proj(output)
        return output


# 测试
X = torch.rand(3, 2, 4)
mask = torch.tril(torch.ones(2, 2))
mask = torch.tensor([
    [1, 1],
    [1, 0]
])
mha = MultiHeadAttention(d_model = 4, num_head = 2)
output = mha(X, mask)
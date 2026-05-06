import torch
import torch.nn as nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # 用一个大线性层生成Q, K, V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()

        # 生成Q, K, V并分成多头
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # reshape为 (B, n_head, T, head_size)
        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        # PyTorch内置的高效注意力计算
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=False  # ⭐ Encoder关键：双向注意力
        )

        # 还原形状
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y)

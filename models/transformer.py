import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# 导入项目配置的超参数
import config

# 设备配置
device = config.device

# -------------------------- 1. 位置编码（正弦余弦） --------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 经典Transformer正弦余弦位置编码
        pe = torch.zeros(block_size, n_embd)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-np.log(10000.0) / n_embd))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# -------------------------- 2. 单头自注意力（带因果掩码） --------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        # 因果掩码：防止看到未来token（Next-token预测核心）
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # 缩放点积注意力
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        # 应用因果掩码
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out

# -------------------------- 3. 多头自注意力 --------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # 拼接多个头的输出
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# -------------------------- 4. 前馈网络 --------------------------
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

# -------------------------- 5. Encoder块（残差+Pre-Norm） --------------------------
class EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 残差连接 + 层归一化
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# -------------------------- 6. 完整 Encoder-only Transformer 模型 --------------------------
class EncoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 令牌嵌入 + 位置编码
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
        self.pos_encoding = PositionalEncoding(config.n_embd, config.block_size, config.dropout)
        # 堆叠Encoder层
        self.blocks = nn.Sequential(*[EncoderBlock(config.n_embd, config.n_head) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        # 输出层（预测下一个token）
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # 嵌入层
        tok_emb = self.token_embedding(idx)
        x = self.pos_encoding(tok_emb)
        # Encoder 推理
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # 计算损失
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    # 文本生成函数（Next-token预测推理）
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # 截断上下文，防止越界
            idx_cond = idx[:, -config.block_size:]
            logits, _ = self(idx_cond)
            # 取最后一个token的预测结果
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # 采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class PositionalEncoding(nn.Module):
    def __init__(self, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(block_size, n_embd)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if n_embd > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        _, t, _ = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


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


class EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class EncoderOnlyTransformer(nn.Module):
    """Causal self-attention language model built from encoder-style blocks."""

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
        self.pos_encoding = PositionalEncoding(
            config.n_embd, config.block_size, config.dropout
        )
        self.blocks = nn.Sequential(
            *[EncoderBlock(config.n_embd, config.n_head) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        _, t = idx.shape
        if t > config.block_size:
            raise ValueError(
                f"Sequence length {t} exceeds block_size {config.block_size}"
            )

        x = self.token_embedding(idx)
        x = self.pos_encoding(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            b, t, c = logits.shape
            loss = F.cross_entropy(logits.reshape(b * t, c), targets.reshape(b * t))
        return logits, loss

    @staticmethod
    def _filter_logits(logits, top_k=None, top_p=None):
        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

        if top_p is not None:
            if not 0 < top_p <= 1:
                raise ValueError("top_p must be in the interval (0, 1]")
            if top_p < 1:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -config.block_size :]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                logits = self._filter_logits(logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        finally:
            self.train(was_training)
        return idx

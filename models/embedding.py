import torch
import torch.nn as nn

class TransformerEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)

        tok_embeds = self.tok_emb(idx)  # 词嵌入
        pos_embeds = self.pos_emb(pos)  # 位置嵌入

        return self.dropout(tok_embeds + pos_embeds)

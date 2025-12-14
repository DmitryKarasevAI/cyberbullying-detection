import torch
from omegaconf import DictConfig
from torch import nn


class AttnMLPBlock(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(cfg.dropout)

        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop1(attn_out)

        return x + self.mlp(self.ln2(x))


class SimpleAttnMLP(nn.Module):
    def __init__(self, vocab_size: int, cfg: DictConfig):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([AttnMLPBlock(cfg) for _ in range(cfg.n_layers)])

        self.ln_out = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)

    def forward(self, tokens: torch.Tensor, pad_mask: torch.Tensor | None = None):
        x = self.drop(self.emb(tokens))

        for block in self.blocks:
            x = block(x, key_padding_mask=pad_mask)

        x = self.ln_out(x)

        if pad_mask is not None:
            keep = (~pad_mask).to(x.dtype)
            denom = keep.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (x * keep[:, :, None]).sum(dim=1) / denom
        else:
            pooled = x.mean(dim=1)

        return self.head(pooled)

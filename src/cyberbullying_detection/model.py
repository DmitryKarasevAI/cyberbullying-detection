import torch
from torch import nn


class AttnMLPBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop1(attn_out)

        return x + self.mlp(self.ln2(x))


class SimpleAttnMLP(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        n_classes: int = 2,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [AttnMLPBlock(d_model, n_heads, d_ff, dropout=dropout) for _ in range(n_layers)]
        )

        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

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

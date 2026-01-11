import torch
from torch import nn
from typing import Optional


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
        )

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.ln_q(latents)
        k = self.ln_kv(context)
        v = k
        attended, _ = self.attn(q, k, v, need_weights=False)
        latents = latents + self.dropout(attended)
        latents = latents + self.ff(latents)
        return latents


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        num_latents: int = 32,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(dim=dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_tokens: (batch, seq_len, dim)
        Returns:
            resampled latents: (batch, num_latents, dim)
        """
        b = image_tokens.shape[0]
        latents = self.latents.unsqueeze(0).expand(b, -1, -1)
        for layer in self.layers:
            latents = layer(latents, image_tokens)
        return latents


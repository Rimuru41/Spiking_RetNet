"""
model.py
--------
SpikingRetNet architecture for text classification.
"""

import math
import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device).floor_() + keep_prob
        return x.div(keep_prob) * random_tensor


# ──────────────────────────────────────────────────────────────────────────────
# Spiking Retention
# ──────────────────────────────────────────────────────────────────────────────

class SpikingRetention(nn.Module):
    def __init__(self, dim, num_heads=8, backend='torch', dropout=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim       = dim
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # Q
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.q_bn   = nn.BatchNorm1d(dim)
        self.q_lif  = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        # K
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.k_bn   = nn.BatchNorm1d(dim)
        self.k_lif  = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        # V
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.v_bn   = nn.BatchNorm1d(dim)
        self.v_lif  = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        # Output
        self.proj_out = nn.Linear(dim, dim)
        self.bn_out   = nn.BatchNorm1d(dim)
        self.lif_out  = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.attn_drop = nn.Dropout(dropout)

        # Per-head causal decay rates
        gamma = 1.0 - 2.0 ** (-5.0 - torch.arange(0, num_heads, dtype=torch.float32))
        self.register_buffer('gamma', gamma)   # [H]

    def _causal_decay(self, N, device):
        """D[h,i,j] = gamma[h]^(i-j)  if i>=j  else 0.  Shape: [H, N, N]"""
        i    = torch.arange(N, device=device).unsqueeze(1).float()
        j    = torch.arange(N, device=device).unsqueeze(0).float()
        diff = (i - j).clamp(min=0)
        mask = (i >= j).float()
        return (self.gamma.view(-1, 1, 1) ** diff) * mask   # [H, N, N]

    def _bn2d(self, bn, x_flat):
        """Apply BN to [T*B, N, C] by treating C as the feature dim."""
        return bn(x_flat.transpose(-1, -2)).transpose(-1, -2)

    def forward(self, x):
        T, B, N, C = x.shape
        xf = x.flatten(0, 1)   # [T*B, N, C]

        q = self.q_lif(self._bn2d(self.q_bn, self.q_proj(xf)).reshape(T, B, N, C))
        k = self.k_lif(self._bn2d(self.k_bn, self.k_proj(xf)).reshape(T, B, N, C))
        v = self.v_lif(self._bn2d(self.v_bn, self.v_proj(xf)).reshape(T, B, N, C))

        # [T, B, H, N, head_dim]
        q = q.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = k.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = v.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        D    = self._causal_decay(N, x.device)                         # [H, N, N]
        attn = (q @ k.transpose(-1, -2)) * self.scale                  # [T,B,H,N,N]
        attn = self.attn_drop(attn * D.unsqueeze(0).unsqueeze(0))

        out = (attn @ v).permute(0, 1, 3, 2, 4).reshape(T * B, N, C).contiguous()
        out = self._bn2d(self.bn_out, self.proj_out(out))
        return self.lif_out(out.reshape(T, B, N, C))


# ──────────────────────────────────────────────────────────────────────────────
# MLP
# ──────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0., backend='torch'):
        super().__init__()
        hidden = hidden_features or in_features

        self.fc1  = nn.Linear(in_features, hidden)
        self.bn1  = nn.BatchNorm1d(hidden)
        self.lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.fc2  = nn.Linear(hidden, in_features)
        self.bn2  = nn.BatchNorm1d(in_features)
        self.lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.drop = nn.Dropout(drop)

    def _bn2d(self, bn, x_flat):
        return bn(x_flat.transpose(-1, -2)).transpose(-1, -2)

    def forward(self, x):
        T, B, N, C = x.shape

        x = self.lif1(self._bn2d(self.bn1, self.fc1(x.flatten(0, 1))).reshape(T, B, N, -1))
        x = self.drop(x)
        x = self.lif2(self._bn2d(self.bn2, self.fc2(x.flatten(0, 1))).reshape(T, B, N, C))
        return self.drop(x)


# ──────────────────────────────────────────────────────────────────────────────
# Block
# ──────────────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., drop_path=0., backend='torch'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn  = SpikingRetention(dim, num_heads, backend, drop)
        self.mlp   = MLP(dim, int(dim * mlp_ratio), drop, backend)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        T, B, N, C = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x.flatten(0, 1)).reshape(T, B, N, C)))
        x = x + self.drop_path(self.mlp(self.norm2(x.flatten(0, 1)).reshape(T, B, N, C)))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Main Model
# ──────────────────────────────────────────────────────────────────────────────

class SpikingRetNetText(nn.Module):
    def __init__(
        self,
        vocab_size=30522,
        max_len=512,
        num_classes=2,
        embed_dims=256,
        num_heads=8,
        mlp_ratios=4,
        depths=2,
        T=4,
        backend='torch',
        dropout=0.0,
        token_drop_prob=0.0,   # kept for API compat (used in training loop)
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.T = T

        self.embedding  = nn.Embedding(vocab_size, embed_dims, padding_idx=0)
        self.pos_embed  = nn.Parameter(torch.zeros(1, max_len, embed_dims))
        trunc_normal_(self.pos_embed, std=0.02)

        self.embed_drop = nn.Dropout(dropout)
        self.embed_lif  = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        self.blocks = nn.ModuleList([
            Block(embed_dims, num_heads, mlp_ratios, dropout, dpr[i], backend)
            for i in range(depths)
        ])

        self.norm = nn.LayerNorm(embed_dims)
        self.head = nn.Linear(embed_dims, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            trunc_normal_(m.weight, std=0.02)

    def forward(self, input_ids, attention_mask):
        # Embed → [B, N, C]
        x = self.embed_drop(
            self.embedding(input_ids) + self.pos_embed[:, :input_ids.size(1), :]
        )
        # Repeat over T → spike-encode → [T, B, N, C]
        x = self.embed_lif(x.unsqueeze(0).repeat(self.T, 1, 1, 1))

        for blk in self.blocks:
            x = blk(x)

        # Time average → [B, N, C]
        x = x.mean(dim=0)

        # Masked mean pooling (ignores PAD tokens)
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return self.head(self.norm(pooled))
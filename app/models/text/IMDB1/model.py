import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x): return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l, u = norm_cdf((a - mean) / std), norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1).erfinv_().mul_(std * math.sqrt(2.)).add_(mean).clamp_(min=a, max=b)
        return tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training: return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device).floor_()
        return x.div(keep_prob) * random_tensor

class SpikingRetention(nn.Module):
    def __init__(self, dim, num_heads=8, backend='torch', dropout=0.):
        super().__init__()
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Independent LIF nodes to prevent shape-mismatch errors
        self.q_proj, self.q_bn = nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.k_proj, self.k_bn = nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.v_proj, self.v_bn = nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.proj_out, self.bn_out = nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        self.lif_out = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        
        self.attn_dropout = nn.Dropout(dropout)
        gamma = 1.0 - 2.0 ** (-5.0 - torch.arange(0, num_heads, dtype=torch.float32))
        self.register_buffer('gamma', gamma)

    def forward(self, x):
        T, B, N, C = x.shape
        x_f = x.flatten(0, 1) # [T*B, N, C]

        # Binary Spike Projections with BN Dimension Fix
        q = self.q_lif(self.q_bn(self.q_proj(x_f).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C))
        k = self.k_lif(self.k_bn(self.k_proj(x_f).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C))
        v = self.v_lif(self.v_bn(self.v_proj(x_f).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C))

        q = q.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = k.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = v.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        # Causal RetNet Decay
        n, m = torch.arange(N, device=x.device).unsqueeze(1), torch.arange(N, device=x.device).unsqueeze(0)
        D = (self.gamma.view(-1, 1, 1) ** (n - m)) * (n >= m).float()
        
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = self.attn_dropout(attn * D.unsqueeze(0).unsqueeze(0))

        out = (attn @ v).transpose(2, 3).reshape(T*B, N, C).contiguous()
        out = self.bn_out(self.proj_out(out).transpose(-1, -2)).transpose(-1, -2)
        return self.lif_out(out.reshape(T, B, N, C))

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0., backend='torch'):
        super().__init__()
        hidden = hidden_features or in_features
        self.fc1, self.bn1 = nn.Linear(in_features, hidden), nn.BatchNorm1d(hidden)
        self.lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.fc2, self.bn2 = nn.Linear(hidden, in_features), nn.BatchNorm1d(in_features)
        self.lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        T, B, N, C = x.shape
        # Layer 1
        x = self.fc1(x.flatten(0, 1)).transpose(-1, -2)
        x = self.lif1(self.bn1(x).transpose(-1, -2).reshape(T, B, N, -1))
        x = self.dropout(x)
        # Layer 2
        x = self.fc2(x.flatten(0, 1)).transpose(-1, -2)
        x = self.lif2(self.bn2(x).transpose(-1, -2).reshape(T, B, N, -1))
        return self.dropout(x)

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., drop_path=0., backend='torch'):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = SpikingRetention(dim, num_heads, backend, drop)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop, backend)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        T, B, N, C = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x.flatten(0, 1)).reshape(T, B, N, C)))
        x = x + self.drop_path(self.mlp(self.norm2(x.flatten(0, 1)).reshape(T, B, N, C)))
        return x

class SpikingRetNetText(nn.Module):
    def __init__(self, vocab_size=30522, max_len=512, num_classes=2, embed_dims=256, 
                 num_heads=8, mlp_ratios=4, depths=2, T=4, backend='torch', 
                 dropout=0.0, token_drop_prob=0.0, drop_path_rate=0.0): 
        super().__init__()
        self.T = T
        self.embedding = nn.Embedding(vocab_size, embed_dims)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dims))
        trunc_normal_(self.pos_embed, std=.02)
        self.embed_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)] 
        self.blocks = nn.ModuleList([Block(embed_dims, num_heads, mlp_ratios, dropout, dpr[i], backend) for i in range(depths)])
        self.head = nn.Linear(embed_dims, num_classes)

    def forward(self, x, attention_mask):
        # Embed and repeat for T steps
        x = self.embed_lif((self.embedding(x) + self.pos_embed[:, :x.size(1), :]).unsqueeze(0).repeat(self.T, 1, 1, 1))
        
        for blk in self.blocks:
            x = blk(x)

        # Mask-Aware Pooling: Averaging only non-padding tokens
        x = x.mean(0) # Average over Time -> [B, N, C]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = torch.sum(x * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return self.head(pooled)
    
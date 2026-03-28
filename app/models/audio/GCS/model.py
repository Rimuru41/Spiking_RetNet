import torch
import torch.nn as nn
import math
import numpy as np
from spikingjelly.activation_based import neuron, surrogate

# Fix for NumPy compatibility in older environments
np.int = int
np.float = float
np.bool = bool

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x): return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l, u = norm_cdf((a - mean) / std), norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1).erfinv_().mul_(std * math.sqrt(2.)).add_(mean).clamp_(min=a, max=b)
        return tensor

class SpikingRetention(nn.Module):
    def __init__(self, dim, num_heads=8, T=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        
        # Decay factor gamma
        gamma = 1.0 - 2.0 ** (-5.0 - torch.arange(0, num_heads).float())
        self.register_buffer('gamma', gamma)
        
        # Spiking Neuron
        self.lif = neuron.LIFNode(
            tau=2.0, 
            surrogate_function=surrogate.ATan(), 
            detach_reset=True, 
            step_mode='m', 
            backend='torch'
        )
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        T, B, N, C = x.shape
        q = self.q_proj(x).reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = self.k_proj(x).reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = self.v_proj(x).reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        # Bidirectional Decay Logic
        n_idx = torch.arange(N, device=x.device)
        dist = (n_idx.unsqueeze(0) - n_idx.unsqueeze(1)).abs() 
        decay = self.gamma.view(-1, 1, 1) ** dist
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn * decay.unsqueeze(0).unsqueeze(0)
        
        out = (attn @ v).permute(0, 1, 3, 2, 4).reshape(T, B, N, C)
        
        # Spiking activation and BN
        out = self.lif(out)
        out_flat = out.flatten(0, 1).transpose(1, 2)
        return self.bn(out_flat).transpose(1, 2).reshape(T, B, N, C)

class SRN_KWS(nn.Module):
    def __init__(self, num_classes=12, embed_dims=128, T=4):
        super().__init__()
        self.T = T
        self.conv = nn.Conv1d(40, embed_dims, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(embed_dims)
        self.lif = neuron.LIFNode(
            tau=2.0, 
            surrogate_function=surrogate.ATan(), 
            detach_reset=True, 
            step_mode='m', 
            backend='torch'
        )
        self.retention = SpikingRetention(embed_dims, num_heads=8, T=T)
        self.head = nn.Linear(embed_dims, num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # x shape: (Batch, Channels, Time_Steps) -> (B, 40, 101)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1) # Expand for Spiking T
        T, B, C, N = x.shape
        
        # Pre-processing Conv + Spiking
        x = self.bn(self.conv(x.flatten(0, 1))).reshape(T, B, -1, N)
        x = self.lif(x).permute(0, 1, 3, 2) # (T, B, N, C)
        
        # Spiking Retention Block
        x = x + self.retention(x)
        
        # Global Pooling across Spiking Time and Sequence Length
        return self.head(x.mean(dim=(0, 2)))
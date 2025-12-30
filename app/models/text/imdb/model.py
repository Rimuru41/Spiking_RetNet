# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from spikingjelly.clock_driven.neuron import MultiStepLIFNode

# # ==========================================
# # HELPER FUNCTIONS
# # ==========================================
# def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
#     def norm_cdf(x):
#         return (1. + math.erf(x / math.sqrt(2.))) / 2.
#     with torch.no_grad():
#         l = norm_cdf((a - mean) / std)
#         u = norm_cdf((b - mean) / std)
#         tensor.uniform_(2 * l - 1, 2 * u - 1)
#         tensor.erfinv_()
#         tensor.mul_(std * math.sqrt(2.))
#         tensor.add_(mean)
#         tensor.clamp_(min=a, max=b)
#         return tensor

# class DropPath(nn.Module):
#     """Drop paths (Stochastic Depth) per sample."""
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob

#     def forward(self, x):
#         if self.drop_prob == 0. or not self.training:
#             return x
#         keep_prob = 1 - self.drop_prob
#         shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#         random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#         random_tensor.floor_()
#         return x.div(keep_prob) * random_tensor

# # ==========================================
# # MODEL COMPONENTS
# # ==========================================

# class MLP(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., backend='torch'):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1_linear = nn.Linear(in_features, hidden_features)
#         self.fc1_bn = nn.BatchNorm1d(hidden_features)
#         self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

#         self.fc2_linear = nn.Linear(hidden_features, out_features)
#         self.fc2_bn = nn.BatchNorm1d(out_features)
#         self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
#         self.dropout = nn.Dropout(drop)
#         self.c_hidden = hidden_features

#     def forward(self, x):
#         T, B, N, C = x.shape
#         x_flat = x.flatten(0, 1)
        
#         x = self.fc1_linear(x_flat)
#         x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous()
#         x = self.fc1_lif(x)
#         x = self.dropout(x)

#         x = self.fc2_linear(x.flatten(0, 1))
#         x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
#         x = self.fc2_lif(x)
#         x = self.dropout(x)
#         return x

# class SpikingRetention(nn.Module):
#     def __init__(self, dim, num_heads=8, bidirectional=True, backend='torch', dropout=0.):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#         self.bidirectional = bidirectional

#         self.q_linear = nn.Linear(dim, dim); self.q_bn = nn.BatchNorm1d(dim)
#         self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

#         self.k_linear = nn.Linear(dim, dim); self.k_bn = nn.BatchNorm1d(dim)
#         self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

#         self.v_linear = nn.Linear(dim, dim); self.v_bn = nn.BatchNorm1d(dim)
#         self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

#         self.proj_linear = nn.Linear(dim, dim); self.proj_bn = nn.BatchNorm1d(dim)
#         self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
#         self.retention_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=backend)
        
#         self.attn_dropout = nn.Dropout(dropout)
#         gamma = 1.0 - 2.0 ** (-5.0 - torch.arange(0, num_heads, dtype=torch.float32))
#         self.register_buffer('gamma', gamma)

#     def get_decay_mask(self, N, device):
#         index = torch.arange(N, device=device).unsqueeze(0)
#         mask = index.unsqueeze(2) - index.unsqueeze(1)
#         gamma_ = self.gamma.view(-1, 1, 1).to(device)
#         D = gamma_ ** (mask.abs()) if self.bidirectional else torch.tril(gamma_ ** mask)
#         return D

#     def forward(self, x):
#         T, B, N, C = x.shape
#         x_flat = x.flatten(0, 1)

#         q = self.q_lif(self.q_bn(self.q_linear(x_flat).transpose(-1,-2)).transpose(-1,-2).reshape(T,B,N,C))
#         k = self.k_lif(self.k_bn(self.k_linear(x_flat).transpose(-1,-2)).transpose(-1,-2).reshape(T,B,N,C))
#         v = self.v_lif(self.v_bn(self.v_linear(x_flat).transpose(-1,-2)).transpose(-1,-2).reshape(T,B,N,C))

#         q = q.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
#         k = k.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
#         v = v.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

#         attn_score = (q @ k.transpose(-2, -1)) * self.scale
#         D = self.get_decay_mask(N, x.device)
#         attn_score = attn_score * D.unsqueeze(0).unsqueeze(0)
#         attn_score = self.attn_dropout(attn_score)

#         x = attn_score @ v
#         x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
#         x = self.retention_lif(x)
#         x = self.proj_lif(self.proj_bn(self.proj_linear(x.flatten(0, 1)).transpose(-1,-2)).transpose(-1,-2).reshape(T,B,N,C))
#         return x

# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., drop_path=0., backend='torch'):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = SpikingRetention(dim, num_heads=num_heads, backend=backend, dropout=drop)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop, backend=backend)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         x = x + self.drop_path(self.attn(x))
#         x = x + self.drop_path(self.mlp(x))
#         return x

# class SpikingTextEmbedding(nn.Module):
#     def __init__(self, vocab_size, embed_dim, max_len, T=4, backend='torch', dropout=0.0, token_drop_prob=0.0):
#         super().__init__()
#         self.T = T
#         self.token_drop_prob = token_drop_prob
#         self.MASK_TOKEN_ID = 103 
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
#         trunc_normal_(self.pos_embed, std=.02)
#         self.norm = nn.LayerNorm(embed_dim)
#         self.dropout = nn.Dropout(p=dropout) 
#         self.lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

#     def forward(self, x):
#         x = self.embedding(x)
#         x = x + self.pos_embed[:, :x.size(1), :] 
#         x = self.norm(x)
#         x = self.dropout(x)
#         x = x.unsqueeze(0).repeat(self.T, 1, 1, 1) 
#         x = self.lif(x)
#         return x

# # class SpikingRetNetText(nn.Module):
# #     def __init__(self, vocab_size=30522, max_len=512, num_classes=2, embed_dims=256, num_heads=8, mlp_ratios=4, depths=2, T=4, backend='torch', dropout=0.0, token_drop_prob=0.0, drop_path_rate=0.0): 
# #         super().__init__()
# #         self.T = T
# #         self.text_embed = SpikingTextEmbedding(vocab_size, embed_dims, max_len, T, backend, dropout=dropout, token_drop_prob=token_drop_prob)
# #         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)] 
# #         self.blocks = nn.ModuleList([Block(dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, drop=dropout, drop_path=dpr[i], backend=backend) for i in range(depths)])
# #         self.head = nn.Linear(embed_dims, num_classes)
# #         self.apply(self._init_weights)

# #     def _init_weights(self, m):
# #         if isinstance(m, nn.Linear):
# #             trunc_normal_(m.weight, std=.02)
# #             if m.bias is not None: nn.init.constant_(m.bias, 0)
# #         elif isinstance(m, nn.LayerNorm):
# #             nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

# #     def forward(self, x):
# #         x = self.text_embed(x) 
# #         for blk in self.blocks: x = blk(x)
# #         x = x.mean(2).mean(0) 
# #         x = self.head(x)
# #         return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

# --- HELPER FUNCTIONS ---
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
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training: return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# --- MODEL COMPONENTS ---
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., backend='torch'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.dropout = nn.Dropout(drop)
        self.c_hidden = hidden_features
    def forward(self, x):
        T, B, N, C = x.shape
        x_flat = x.flatten(0, 1)
        x = self.fc1_linear(x_flat)
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous()
        x = self.fc1_lif(x)
        x = self.dropout(x)
        x = self.fc2_linear(x.flatten(0, 1))
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        x = self.fc2_lif(x)
        x = self.dropout(x)
        return x

class SpikingRetention(nn.Module):
    def __init__(self, dim, num_heads=8, bidirectional=True, backend='torch', dropout=0.):
        super().__init__()
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.bidirectional = bidirectional
        self.q_linear = nn.Linear(dim, dim); self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.k_linear = nn.Linear(dim, dim); self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.v_linear = nn.Linear(dim, dim); self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.proj_linear = nn.Linear(dim, dim); self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.retention_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=backend)
        self.attn_dropout = nn.Dropout(dropout)
        gamma = 1.0 - 2.0 ** (-5.0 - torch.arange(0, num_heads, dtype=torch.float32))
        self.register_buffer('gamma', gamma)
    def get_decay_mask(self, N, device):
        index = torch.arange(N, device=device).unsqueeze(0)
        mask = index.unsqueeze(2) - index.unsqueeze(1)
        gamma_ = self.gamma.view(-1, 1, 1).to(device)
        return gamma_ ** (mask.abs()) if self.bidirectional else torch.tril(gamma_ ** mask)
    def forward(self, x):
        T, B, N, C = x.shape
        x_flat = x.flatten(0, 1)
        q = self.q_lif(self.q_bn(self.q_linear(x_flat).transpose(-1,-2)).transpose(-1,-2).reshape(T,B,N,C))
        k = self.k_lif(self.k_bn(self.k_linear(x_flat).transpose(-1,-2)).transpose(-1,-2).reshape(T,B,N,C))
        v = self.v_lif(self.v_bn(self.v_linear(x_flat).transpose(-1,-2)).transpose(-1,-2).reshape(T,B,N,C))
        q = q.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = k.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = v.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        attn_score = (q @ k.transpose(-2, -1)) * self.scale
        D = self.get_decay_mask(N, x.device)
        attn_score = attn_score * D.unsqueeze(0).unsqueeze(0)
        attn_score = self.attn_dropout(attn_score)
        x = attn_score @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.retention_lif(x)
        x = self.proj_lif(self.proj_bn(self.proj_linear(x.flatten(0, 1)).transpose(-1,-2)).transpose(-1,-2).reshape(T,B,N,C))
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., drop_path=0., backend='torch'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpikingRetention(dim, num_heads=num_heads, backend=backend, dropout=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop, backend=backend)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x

class SpikingTextEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, T=4, backend='torch', dropout=0.0):
        super().__init__()
        self.T = T
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.norm(x)
        x = self.dropout(x)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1)
        return self.lif(x)

class SpikingRetNetText(nn.Module):
    def __init__(self, vocab_size=30522, max_len=512, num_classes=2, embed_dims=256, num_heads=8, mlp_ratios=4, depths=2, T=4, backend='torch', dropout=0.0, token_drop_prob=0.0, drop_path_rate=0.0):
        super().__init__()
        self.T = T
        self.text_embed = SpikingTextEmbedding(vocab_size, embed_dims, max_len, T, backend, dropout=dropout)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        self.blocks = nn.ModuleList([Block(dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, drop=dropout, drop_path=dpr[i], backend=backend) for i in range(depths)])
        self.head = nn.Linear(embed_dims, num_classes)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
    def forward(self, x, mask=None):
        x = self.text_embed(x)
        for blk in self.blocks: x = blk(x)
        if mask is not None:
            # FIX: Masked Pooling to prevent 50.6% confidence
            m = mask.unsqueeze(0).unsqueeze(-1) # [1, B, N, 1]
            x = (x * m).sum(2) / m.sum(2).clamp(min=1)
        else:
            x = x.mean(2)
        x = x.mean(0)
        return self.head(x)



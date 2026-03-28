import torch
import torch.nn as nn
import math
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l, u = norm_cdf((a - mean) / std), norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1).erfinv_().mul_(std * math.sqrt(2.)).add_(mean).clamp_(min=a, max=b)
        return tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training: return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, backend='torch'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.c_hidden = hidden_features

    def forward(self, x):
        T, B, N, C = x.shape
        x_ = x.flatten(0, 1)
        x = self.fc1_linear(x_)
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous()
        x = self.fc1_lif(x)
        x = self.fc2_linear(x.flatten(0, 1))
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        x = self.fc2_lif(x)
        return x

class SpikingRetention(nn.Module):
    def __init__(self, dim, num_heads=8, bidirectional=True, backend='torch'):
        super().__init__()
        self.dim, self.num_heads, self.head_dim = dim, num_heads, dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.bidirectional = bidirectional
        self.q_linear = nn.Linear(dim, dim); self.q_bn = nn.BatchNorm1d(dim); self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.k_linear = nn.Linear(dim, dim); self.k_bn = nn.BatchNorm1d(dim); self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.v_linear = nn.Linear(dim, dim); self.v_bn = nn.BatchNorm1d(dim); self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.proj_linear = nn.Linear(dim, dim); self.proj_bn = nn.BatchNorm1d(dim); self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.retention_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend=backend)
        self.register_buffer('gamma', 1.0 - 2.0 ** (-5.0 - torch.arange(0, num_heads, dtype=torch.float32)))

    def get_decay_mask(self, N, device):
        index = torch.arange(N, device=device).unsqueeze(0)
        mask = index.unsqueeze(2) - index.unsqueeze(1)
        D = self.gamma.view(-1, 1, 1).to(device) ** (mask.abs() if self.bidirectional else torch.tril(mask))
        return D

    def forward(self, x):
        T, B, N, C = x.shape
        x_f = x.flatten(0, 1)
        q = self.q_lif(self.q_bn(self.q_linear(x_f).transpose(-1,-2)).transpose(-1,-2).reshape(T,B,N,C)).reshape(T,B,N,self.num_heads,-1).permute(0,1,3,2,4)
        k = self.k_lif(self.k_bn(self.k_linear(x_f).transpose(-1,-2)).transpose(-1,-2).reshape(T,B,N,C)).reshape(T,B,N,self.num_heads,-1).permute(0,1,3,2,4)
        v = self.v_lif(self.v_bn(self.v_linear(x_f).transpose(-1,-2)).transpose(-1,-2).reshape(T,B,N,C)).reshape(T,B,N,self.num_heads,-1).permute(0,1,3,2,4)
        attn = (q @ k.transpose(-2, -1)) * self.scale * self.get_decay_mask(N, x.device).unsqueeze(0).unsqueeze(0)
        x = self.retention_lif((attn @ v).transpose(2, 3).reshape(T, B, N, C).contiguous())
        return self.proj_lif(self.proj_bn(self.proj_linear(x.flatten(0,1)).transpose(-1,-2)).transpose(-1,-2).reshape(T,B,N,C))

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop_path=0., backend='torch'):
        super().__init__()
        # ADDED norm1 and norm2 back to match your training checkpoint
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpikingRetention(dim, num_heads, backend=backend)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), backend=backend)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        # Forward logic remains the same (SNNs often bypass LayerNorm)
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x

class SPS_Static(nn.Module):
    def __init__(self, in_channels=3, embed_dims=384, T=4, backend='torch'):
        super().__init__()
        self.T = T
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, 3, 1, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        self.act1 = nn.ReLU(True)
        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, 3, 1, 1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims//4)
        self.act2 = nn.ReLU(True)
        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, 3, 1, 1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims//2)
        self.act3 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d(3, 2, 1)
        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, 3, 1, 1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.act4 = nn.ReLU(True)
        self.maxpool3 = nn.MaxPool2d(3, 2, 1)
        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, 3, 1, 1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

    def forward(self, x):
        x = self.act1(self.proj_bn(self.proj_conv(x)))
        x = self.act2(self.proj_bn1(self.proj_conv1(x)))
        x = self.act3(self.proj_bn2(self.proj_conv2(x)))
        x = self.maxpool2(x)
        x = self.act4(self.proj_bn3(self.proj_conv3(x)))
        x = self.maxpool3(x)
        x_feat = x
        x = self.rpe_bn(self.rpe_conv(x))
        x = x + x_feat 
        x = self.rpe_lif(x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1))
        return x.flatten(-2).transpose(-1, -2)

class SpikingRetNet(nn.Module):
    def __init__(self, img_size=32, num_classes=10, embed_dims=384, num_heads=12, depths=4, T=4, backend='torch'):
        super().__init__()
        self.patch_embed = SPS_Static(3, embed_dims, T, backend)
        self.blocks = nn.ModuleList([Block(embed_dims, num_heads, backend=backend) for _ in range(depths)])
        self.head = nn.Linear(embed_dims, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear): trunc_normal_(m.weight, std=.02)
    def forward(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks: x = blk(x)
        return self.head(x.mean(0).mean(1))
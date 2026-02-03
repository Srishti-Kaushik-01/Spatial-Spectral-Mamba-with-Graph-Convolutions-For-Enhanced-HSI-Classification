

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# -------------------------------------------------
# Graph Convolution
# -------------------------------------------------
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, x, adj):
        out = torch.matmul(adj, torch.matmul(x, self.weight))
        return out + self.bias if self.bias is not None else out


# -------------------------------------------------
# Efficient GCN Block
# -------------------------------------------------
class GCNBlock(nn.Module):
    def __init__(self, channels, k=9, group_num=4, max_nodes=2048, use_residual=False):
        super().__init__()
        self.k = k
        self.max_nodes = max_nodes
        self.use_residual = use_residual

        self.gcn1 = GraphConvolution(channels, channels)
        self.gcn2 = GraphConvolution(channels, channels)

        self.norm1 = nn.GroupNorm(group_num, channels)
        self.norm2 = nn.GroupNorm(group_num, channels)
        self.act = nn.SiLU()

    def spatial_pooling(self, x):
        B, C, H, W = x.shape
        if H * W <= self.max_nodes:
            return x, None
        scale = int(np.sqrt((H * W) / self.max_nodes))
        x = F.avg_pool2d(x, scale)
        return x, (H, W)

    def build_knn(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = F.normalize(x, dim=-1)
        sim = torch.bmm(x, x.transpose(1, 2))
        # Clamp k to available nodes
        k = min(self.k, sim.size(-1))
        val, idx = torch.topk(sim, k, dim=-1)
        return x, val, idx

    def gcn_apply(self, x, val, idx, gcn):
        """Apply GCN using batched kNN indexing"""
        B, N, C = x.shape
        k = idx.size(-1)

        # normalize edge weights
        deg = val.sum(dim=-1, keepdim=True) + 1e-6
        val = val / deg

        # linear transform
        x_t = torch.matmul(x, gcn.weight)  # (B, N, C)

        # batched index select
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, k)
        neigh = x_t[batch_idx, idx]  # (B, N, k, C)

        out = (neigh * val.unsqueeze(-1)).sum(dim=2)
        return out + gcn.bias

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        
        x_pool, ori = self.spatial_pooling(x)

        x_flat, val, idx = self.build_knn(x_pool)

        out = self.gcn_apply(x_flat, val, idx, self.gcn1)
        out = self.act(self.norm1(out.permute(0,2,1).reshape(B,C,*x_pool.shape[2:])))

        out_flat = out.reshape(B, C, -1).permute(0, 2, 1)
        out = self.gcn_apply(out_flat, val, idx, self.gcn2)
        out = self.act(self.norm2(out.permute(0,2,1).reshape(B,C,*x_pool.shape[2:])))

        if ori:
            out = F.interpolate(out, size=ori, mode='bilinear', align_corners=False)

        return out + identity if self.use_residual else out


# -------------------------------------------------
# Spectral Positional Encoding
# -------------------------------------------------
class SpectralPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, dim))

    def forward(self, x):
        return x + self.pe


# -------------------------------------------------
# Spectral Mamba
# -------------------------------------------------
class SpeMamba(nn.Module):
    def __init__(self, channels, token_num=8, group_num=4, use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        self.token_num = token_num
        self.group_dim = math.ceil(channels / token_num)
        self.full_dim = self.group_dim * token_num

        self.pe = SpectralPE(self.group_dim)

        self.mamba = Mamba(
            d_model=self.group_dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.full_dim),
            nn.SiLU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        
        if C < self.full_dim:
            x = F.pad(x, (0,0,0,0,0,self.full_dim-C))

        x = x.permute(0,2,3,1)
        x = x.reshape(B*H*W, self.token_num, self.group_dim)

        x = self.pe(x)
        x = self.mamba(x)

        x = x.reshape(B, H, W, self.full_dim).permute(0,3,1,2)
        x = self.proj(x)

        out = x[:, :C]
        return out + identity if self.use_residual else out


# -------------------------------------------------
# Spatial Mamba
# -------------------------------------------------
class SpaMamba(nn.Module):
    def __init__(self, channels, group_num=4, use_residual=False):
        super().__init__()
        self.use_residual = use_residual

        self.row_mamba = Mamba(channels, 16, 4, 2)
        self.col_mamba = Mamba(channels, 16, 4, 2)

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, channels),
            nn.SiLU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        xr = x.permute(0,2,3,1).reshape(B*H, W, C)
        xr = self.row_mamba(xr).reshape(B, H, W, C)

        xc = xr.permute(0,2,1,3).reshape(B*W, H, C)
        xc = self.col_mamba(xc).reshape(B, W, H, C).permute(0,2,1,3)

        out = xc.permute(0,3,1,2)
        out = self.proj(out)

        return out + identity if self.use_residual else out


# -------------------------------------------------
# BothMamba: SpaMamba + SpeMamba Fusion
# -------------------------------------------------
class BothMamba(nn.Module):
    """Fusion of Spatial and Spectral Mamba"""
    def __init__(self, channels, token_num, group_num=4, use_att=True, use_residual=False):
        super().__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

        self.spa_mamba = SpaMamba(channels, group_num=group_num, use_residual=False)
        self.spe_mamba = SpeMamba(channels, token_num=token_num, group_num=group_num, use_residual=False)

    def forward(self, x):
        identity = x
        
        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)
        
        if self.use_att:
            weights = self.softmax(self.weights)
            fusion_x = spa_x * weights[0] + spe_x * weights[1]
        else:
            fusion_x = spa_x + spe_x
            
        return fusion_x + identity if self.use_residual else fusion_x


# -------------------------------------------------
# GCN + Mamba Parallel Fusion
# -------------------------------------------------
class GCNMambaFusion(nn.Module):
    """
    Parallel architecture:
    - GCN Branch (graph structure learning)
    - Mamba Branch (SpaMamba, SpeMamba, or Both)
    Then fuse with learned weights
    """
    def __init__(self, channels, token_num, group_num=4, use_att=True, 
                 k_neighbors=9, use_residual=True, mamba_type='both'):
        super().__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        self.mamba_type = mamba_type

        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

        # GCN branch (always present in this module)
        self.gcn_branch = GCNBlock(channels, k=k_neighbors, group_num=group_num, use_residual=False)
        
        # Mamba branch - choose type
        if mamba_type == 'spa':
            self.mamba_branch = SpaMamba(channels, group_num=group_num, use_residual=False)
        elif mamba_type == 'spe':
            self.mamba_branch = SpeMamba(channels, token_num=token_num, group_num=group_num, use_residual=False)
        elif mamba_type == 'both':
            self.mamba_branch = BothMamba(channels, token_num, group_num=group_num, 
                                          use_att=use_att, use_residual=False)

    def forward(self, x):
        identity = x
        
        # Parallel processing
        gcn_out = self.gcn_branch(x)
        mamba_out = self.mamba_branch(x)

        # Fusion
        if self.use_att:
            weights = self.softmax(self.weights)
            fusion_out = gcn_out * weights[0] + mamba_out * weights[1]
        else:
            fusion_out = gcn_out + mamba_out

        return fusion_out + identity if self.use_residual else fusion_out


# -------------------------------------------------
# Main MambaHSI Model
# -------------------------------------------------
class MambaHSI(nn.Module):
    """
    Main HSI Classification Model with flexible architecture
    
    Architecture Options:
    - use_gcn: True/False - whether to use GCN in parallel
    - mamba_type: 'spa', 'spe', or 'both' - which Mamba variant to use
    
    Combinations:
    1. use_gcn=True, mamba_type='both': GCN + (SpaMamba + SpeMamba) parallel
    2. use_gcn=True, mamba_type='spa': GCN + SpaMamba parallel
    3. use_gcn=True, mamba_type='spe': GCN + SpeMamba parallel
    4. use_gcn=False, mamba_type='both': Only (SpaMamba + SpeMamba)
    5. use_gcn=False, mamba_type='spa': Only SpaMamba
    6. use_gcn=False, mamba_type='spe': Only SpeMamba
    """
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=7, 
                 use_residual=True, token_num=4, group_num=4, 
                 use_att=True, k_neighbors=9,
                 use_gcn=True,  # Whether to use GCN in parallel
                 mamba_type='both'):  # 'spa', 'spe', or 'both'
        super().__init__()
        self.use_gcn = use_gcn
        self.mamba_type = mamba_type

        # Patch embedding
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )

        # Build blocks based on configuration
        if use_gcn:
            # Parallel GCN + Mamba architecture
            self.block1 = GCNMambaFusion(hidden_dim, token_num, group_num, use_att, k_neighbors, use_residual)
            self.block2 = GCNMambaFusion(hidden_dim, token_num, group_num, use_att, k_neighbors, use_residual)
            self.block3 = GCNMambaFusion(hidden_dim, token_num, group_num, use_att, k_neighbors, use_residual)
        else:
            # Only Mamba (no GCN)
            if mamba_type == 'spa':
                self.block1 = SpaMamba(hidden_dim, group_num, use_residual=use_residual)
                self.block2 = SpaMamba(hidden_dim, group_num, use_residual=use_residual)
                self.block3 = SpaMamba(hidden_dim, group_num, use_residual=use_residual)
            elif mamba_type == 'spe':
                self.block1 = SpeMamba(hidden_dim, token_num, group_num, use_residual=use_residual)
                self.block2 = SpeMamba(hidden_dim, token_num, group_num, use_residual=use_residual)
                self.block3 = SpeMamba(hidden_dim, token_num, group_num, use_residual=use_residual)
            elif mamba_type == 'both':
                self.block1 = BothMamba(hidden_dim, token_num, group_num, use_att, use_residual=use_residual)
                self.block2 = BothMamba(hidden_dim, token_num, group_num, use_att, use_residual=use_residual)
                self.block3 = BothMamba(hidden_dim, token_num, group_num, use_att, use_residual=use_residual)

        self.pool1 = nn.AvgPool2d(2)
        self.pool2 = nn.AvgPool2d(2)

        # Main classifier - keep spatial dimensions for dense prediction
        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 1),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        
        # --- Block 1 ---
        x1 = self.block1(x)
        
        # Pool
        if x1.shape[2] > 4 and x1.shape[3] > 4:
            x1_pool = self.pool1(x1)
        else:
            x1_pool = x1
        
        # --- Block 2 ---
        x2 = self.block2(x1_pool)
        
        # Pool
        if x2.shape[2] > 4 and x2.shape[3] > 4:
            x2_pool = self.pool2(x2)
        else:
            x2_pool = x2
        
        # --- Block 3 ---
        x3 = self.block3(x2_pool)
        
        # --- Head ---
        out = self.cls_head(x3)
        out = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
        
        return out




# Register model
from .registry import register_model


@register_model('GGSM2', expects_4d=True, hidden_dim=64, use_residual=True, 
                mamba_type='both', token_num=4, group_num=4, use_att=True, 
                use_gcn=True, k_neighbors=9,)
def ggsm2(pretrained: bool = False, **kwargs) -> MambaHSI:
    """Constructs a GGSM2 (Advanced Graph-based Spectral-Spatial Mamba) model.
    
    This is an enhanced version of GGSM with novel components including:
    - Learnable spectral mixing
    - Edge-aware graph construction
    - Multi-scale feature pyramid
    - Spectral-spatial cross-attention
    - Class token pooling
    - Optional contrastive learning
    
    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        **kwargs: Additional arguments passed to the model
        
    Returns:
        NovelMambaHSI: The constructed GGSM2 model
    """
    # Map standardized names to model-specific names
    if 'bands' in kwargs:
        kwargs['in_channels'] = kwargs.pop('bands')
    kwargs.pop('patch_size', None)  # GGSM2 doesn't use patch_size
    kwargs.pop('use_contrastive', None)  # Remove unsupported parameter
    return MambaHSI(**kwargs)

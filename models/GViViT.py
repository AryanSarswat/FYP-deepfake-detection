from .util import DropPath
from .layers import GlobalQueryGen, WindowAttention3D
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce, lru_cache

def window_partition(x, window_size):
    """
    Args:
        x: (B, T, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, T, H, W, C = x.shape
    x = x.view(B, T // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows

def window_reverse(windows, window_size, B, T, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, T, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

class GCViTBlock(nn.Module):
    def __init__(self, dim: int, input_resolution: int, num_heads: int, window_size : tuple=(1, 7, 7), mlp_ratio=4., qkv_bias=True, qk_scale: int = None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, attention=WindowAttentionGlobal3D, norm_layer=nn.LayerNorm, layer_scale=None):
        """
        GC-ViT Block
        Args:
            dim (int): _description_
            input_resolution (int): _description_
            num_heads (int): _description_
            window_size (tuple, optional): _description_. Defaults to (1, 7, 7).
            mlp_ratio (_type_, optional): _description_. Defaults to 4..
            qkv_bias (bool, optional): _description_. Defaults to True.
            qk_scale (int, optional): _description_. Defaults to None.
            drop (_type_, optional): _description_. Defaults to 0..
            attn_drop (_type_, optional): _description_. Defaults to 0..
            drop_path (_type_, optional): _description_. Defaults to 0..
            act_layer (_type_, optional): _description_. Defaults to nn.GELU.
            attention (_type_, optional): _description_. Defaults to WindowAttentionGlobal3D.
            norm_layer (_type_, optional): _description_. Defaults to nn.LayerNorm.
            layer_scale (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        
        self.attn = attention(dim=dim, num_heads=num_heads, window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )
        
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int , float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0
        
        inp_width = input_resolution // window_size[1]
        self.num_windows = inp_width * inp_width
        
    def forward(self, x , q_global):
        B, T, C, H, W = x.shape
        shortcut = x
        x = self.norm1(x)
        H_w = H // self.window_size[1]
        W_w = W // self.window_size[2]
        x_windows = window_partition(x, window_size=self.window_size)
        x_windows = x_windows.view(-1, self.window_size[1] * self.window_size[2], C)
        attn_windows = self.attn(x_windows, q_global)
        x = window_reverse(attn_windows, self.window_size, B, T, H_w, W_w)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x
    
class GCViTLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    
class GCViViT(nn.Module):
    """Class for Global Video Vision Transformer.
    """
    def __init__(self, num_frames: int, patch_size: int, in_channels: int, height: int, width: int, dim: int = 192, depth: int = 4, heads: int = 3, head_dims: int = 64, dropout: float = 0., scale_dim: int = 4, spt=False, lsa=False):
        """Constructor for ViViT.

        Args:
            num_frames (int): Number of frames in video
            patch_size (int): Size of patch
            in_channels (int): Number of channels in input
            height (int): Height of input
            width (int): Width of input
            num_classes (int): Number of classes
            dim (int, optional): _description_. Defaults to 192.
            depth (int, optional): _description_. Defaults to 4.
            heads (int, optional): _description_. Defaults to 3.
            head_dims (int, optional): _description_. Defaults to 64.
            dropout (int, optional): _description_. Defaults to 0.
            scale_dim (int, optional): _description_. Defaults to 4.
        """
        super().__init__()
        
        assert height % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        assert width % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (height // patch_size) * (width // patch_size)
        
        self.to_patch_embedding = PatchEmbedding(img_size=height, patch_size=patch_size, in_channels=in_channels, embed_dim=dim) if not spt else ShiftedPatchTokenization(dim=dim, patch_size=patch_size, channels=in_channels)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.spatial_token = nn.Parameter(torch.randn(1, 1, dim))
        self.spatial_transformer = GCViViTransformer(token_dim=dim, depth=depth, head_dims=head_dims, heads=heads, mlp_dim=dim*scale_dim, dropout=dropout, lsa=lsa)
        
        self.temporal_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
        self.temporal_transformer = GCViViTransformer(token_dim=dim, depth=depth, head_dims=head_dims, heads=heads, mlp_dim=dim*scale_dim, dropout=dropout, lsa=lsa)
        
        self.dropout = nn.Dropout(dropout)
        
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        trunc_normal_(self.pos_embedding, std=.02)
        trunc_normal_(self.temporal_embedding, std=.02)
        trunc_normal_(self.spatial_token, std=.02)
        self.apply(self._init_weights)
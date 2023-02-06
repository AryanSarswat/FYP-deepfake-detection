from util import DropPath, trunc_normal_
from layers import GlobalQueryGen, SqueezeExcitation, ReduceSize, PatchEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import reduce, lru_cache

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, B, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        B (int): Batch size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, H, W)
    return x

class FeatureExtract(nn.Module):
    def __init__(self, dim, reduce=True) -> None:
        super(FeatureExtract, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim , kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
            nn.GELU(),
            SqueezeExcitation(dim, dim),
            nn.Conv2d(in_channels=dim, out_channels=dim , kernel_size=1, stride=1, padding=0, bias=False),
        )
        if reduce:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.reduce = reduce
        
    def forward(self, x):
        x = x.contiguous()
        x = x  + self.conv(x)
        if self.reduce:
            x = self.pool(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.window_size = (window_size, window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x , q_global):
        B_ , N , C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        queries, keys, values = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        
        queries = queries * self.scale
        attn = (queries @ keys.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x = (attn @ values).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class WindowAttentionGlobal(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(WindowAttentionGlobal, self).__init__()
        window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads, rounding_mode='floor')
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q_global):
        B_, N, C = x.shape
        B = q_global.shape[0]
        head_dim = C // self.num_heads
        B_dim = B_ // B
        kv = self.qkv(x).chunk(2, dim=-1)
        keys, values = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), kv)
        
        
        temp = q_global.repeat(1, B_dim, 1, 1, 1)
        queries = temp.reshape(B_, self.num_heads, N, head_dim)
        queries = queries * self.scale
        attn = (queries @ keys.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x = (attn @ values).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        
class GCViTBlock(nn.Module):
    def __init__(self, dim: int, input_resolution: int, num_heads: int, window_size : int, mlp_ratio=4., qkv_bias=True, qk_scale: int = None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, attention=WindowAttentionGlobal, norm_layer=nn.LayerNorm, layer_scale=None):
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
        super(GCViTBlock, self).__init__()
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
        
        inp_width = input_resolution // window_size
        self.num_windows = int(inp_width * inp_width)
        
    def forward(self, x , q_global):
        B, C, H, W = x.shape
        shortcut = x


        # Partition into windows
        x_windows = window_partition(x, self.window_size)
        # Convert to tokens
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        # Pre Norm
        x_windows = self.norm1(x_windows)
        
        attn_windows = self.attn(x_windows, q_global)
        
        # Convert back to windows
        x = window_reverse(attn_windows, self.window_size, B, H, W)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x.view(B, H, W, C)
        # MLP
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        x = x.view(B, C, H, W)
        return x
    
class GlobalQueryGenerator(nn.Module):
    def __init__(self, in_dims, input_res, img_res, window_size, num_heads, reduce=True):
        super(GlobalQueryGenerator, self).__init__()
        if input_res == img_res // 4:
            self.to_q_global = nn.Sequential(
                FeatureExtract(in_dims),
                FeatureExtract(in_dims),
                FeatureExtract(in_dims),
            )
        elif input_res == img_res // 8:
            self.to_q_global = nn.Sequential(
                FeatureExtract(in_dims),
                FeatureExtract(in_dims),
            )
        elif input_res == img_res // 16:
            if window_size == input_res:
                self.to_q_global = nn.Sequential(
                    FeatureExtract(in_dims, reduce=False),
                )
            else:
                self.to_q_global = nn.Sequential(
                    FeatureExtract(in_dims),
                )
        elif input_res == img_res // 32:
            self.to_q_global = nn.Sequential(
                FeatureExtract(in_dims, reduce=False),
            )
        
        self.resolution = input_res
        self.num_heads = num_heads
        self.N = window_size * window_size
        self.head_dim = in_dims // num_heads
                
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.to_q_global(x)
        x = x.reshape(B, -1, self.num_heads, self.N, self.head_dim)
        return x

class GCViTLevel(nn.Module):
    def __init__(self, dim, depth, input_resolution, img_resolution, num_heads, window_size, downsample=True, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, layer_scale=None):
        super(GCViTLevel, self).__init__()
        self.blocks = nn.ModuleList([
            GCViTBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attention=WindowAttention if (i % 2 == 0) else WindowAttentionGlobal,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                layer_scale=layer_scale,
                input_resolution=input_resolution)
            for i in range(depth)])
        self.downsample = None if not downsample else ReduceSize(dim=dim, norm_layer=norm_layer)
        self.q_global_generator = GlobalQueryGenerator(dim, input_res=input_resolution, img_res=img_resolution, window_size=window_size, num_heads=num_heads)
        
    def forward(self, x):
        B, C, H, W = x.shape
        q_global = self.q_global_generator(x)
        for blk in self.blocks:
            x = blk(x, q_global)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
    
class GCViT(nn.Module):
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
        
        self.to_patch_embedding = PatchEmbedding(img_size=height, patch_size=patch_size, in_channels=in_channels, embed_dim=dim)
        
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
        
if __name__ == '__main__':
    
    test = torch.randn(2, 64, 56, 56)
    
    layer_test = GCViTLevel(dim=64, depth=4, input_resolution=56, img_resolution=224, num_heads=4, window_size=7, downsample=True, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, layer_scale=None)
    out = layer_test(test)
    
    print(out.shape)
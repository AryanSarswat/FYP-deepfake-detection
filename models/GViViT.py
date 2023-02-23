import torch
import torch.nn as nn
from einops import rearrange, repeat

from .layers import ReduceSize, SqueezeExcitation
from .Transformer import Transformer
from .util import DropPath, trunc_normal_


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
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
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, H, W)
    return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=3, stride=2, padding=1)
        self.conv_down = ReduceSize(dim, reduce=False)
    
    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
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
        x = x.reshape(B, H, W, C)
        x = self.drop_path(self.gamma1 * x)
        x = x.reshape(B, C, H, W)
        x = shortcut + x
        x = x.reshape(B, H, W, C)
        # MLP
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        x = x.reshape(B, C, H, W)
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
    def __init__(self, dim, depths, window_size, mlp_ratio, num_heads, num_classes=0, resolution=224, drop_path_rate=0.2, in_chan=3, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.patch_embed = PatchEmbedding(in_channels=in_chan, dim=dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            self.levels.append(
                GCViTLevel(
                    dim=dim * 2 ** i,
                    depth=depths[i],
                    input_resolution=int(2 ** (-2 - i) * resolution),
                    img_resolution=resolution,
                    num_heads=num_heads[i],
                    window_size=window_size[i],
                    downsample=(i < len(depths) - 1),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    norm_layer=norm_layer,
                    layer_scale=layer_scale))
        self.norm = norm_layer(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, 1) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        for level in self.levels:
            x = level(x)
        
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.avgpool(x)
        x = x.flatten(1)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
class GCViViT(nn.Module):
    """Class for Global Video Vision Transformer.
    """
    def __init__(self, num_frames: int, in_channels, dim: int = 1024, depth: int = 4, heads: int = 3, head_dims: int = 64, dropout: float = 0., scale_dim: int = 4, spt=False, lsa=False):
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
        
        self.spatial_transformer = GCViT(
            depths=GCViT_small_config['depths'],
            num_heads=GCViT_small_config['num_heads'],
            window_size=GCViT_small_config['window_size'],
            dim=GCViT_small_config['dim'],
            mlp_ratio=GCViT_small_config['mlp_ratio'],
            drop_path_rate=GCViT_small_config['drop_path_rate'],
            layer_scale=GCViT_small_config['layer_scale'],
            in_chan=in_channels,
        )
        
        
        self.temporal_embedding = nn.Parameter(torch.randn(1, num_frames + 1, dim))
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(token_dim=dim, depth=depth, head_dims=head_dims, heads=heads, mlp_dim=dim*scale_dim, dropout=dropout, lsa=lsa)
        
        self.dropout = nn.Dropout(dropout)
        
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )
        
        # Initialize weights
        trunc_normal_(self.temporal_embedding, std=.02)

    def forward(self, x):
        B, F, C, H, W = x.shape
        
        # Process spatial dimensions first
        x = x.reshape(B*F, C, H, W)
        x = self.spatial_transformer(x)
        
        # Unfold time
        x = x.reshape(B, F, -1)
        cls_token = repeat(self.temporal_cls_token, '() n d -> b n d', b=B)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.temporal_embedding
        x = self.temporal_transformer(x)
        # Taking only the cls token
        x = x[:, 0]
        x = self.dropout(x)
        
        vectors = x
        
        x = self.head(x)
        return x, vectors
        

GCViT_large_config = {
    'depths' : [3, 4, 19, 5],
    'num_heads' : [6, 12, 24, 48],
    'window_size' : [7, 7, 14, 7],
    'dim' : 192,
    'mlp_ratio' : 2,
    'drop_path_rate' : 0.5,
    'layer_scale' : 1e-5,
}

GCViT_base_config = {
    'depths' : [3, 4, 19, 5],
    'num_heads' : [4, 8, 16, 32],
    'window_size' : [7, 7, 14, 7],
    'dim' : 128,
    'mlp_ratio' : 2,
    'drop_path_rate' : 0.5,
    'layer_scale' : 1e-5,
}

GCViT_small_config = {
    'depths' : [3, 4, 19, 5],
    'num_heads' : [3, 6, 12, 24],
    'window_size' : [7, 7, 14, 7],
    'dim' : 96,
    'mlp_ratio' : 2,
    'drop_path_rate' : 0.3,
    'layer_scale' : 1e-5,
}

GCViT_tiny_config = {
    'depths' : [3, 4, 19, 5],
    'num_heads' : [2, 4, 8, 16],
    'window_size' : [7, 7, 14, 7],
    'dim' : 64,
    'mlp_ratio' : 3,
    'drop_path_rate' : 0.3,
    'layer_scale' : 1e-5,
}


def create_model(num_frames, in_channels, dim, dropout=0, lsa=False):
    model = GCViViT(num_frames=num_frames, in_channels=in_channels, lsa=lsa, dim=dim, dropout=dropout)
    return model

def load_model(path, num_frames, in_channels, dim, lsa=False) :
    model = create_model(num_frames, in_channels, dim, lsa=lsa)
    model.load_state_dict(torch.load(path))
    return model

if __name__ == '__main__':
    test = torch.randn(2, 3, 3, 224, 224)
    gcvit = GCViViT(num_frames=3, in_channels=3)
    #summary(gcvit, (3, 3, 224, 224), device='cpu')
    out = gcvit(test)
    print(out.shape)

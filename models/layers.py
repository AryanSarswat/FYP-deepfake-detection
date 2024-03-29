from collections import OrderedDict
from functools import partial

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from einops.layers.torch import Rearrange
from torch import nn


class ShiftedPatchTokenization(nn.Module):
    def __init__(self, dim: int, patch_size: int, channels=3):
        """
        Module for patch tokenization. taken from https://arxiv.org/pdf/2112.13492.pdf

        Args:
            dim (int): Dimension of patch token.
            patch_size (int): Size of patch.
            channels (int, optional): Number of input channels. Defaults to 3.
        """
        super(ShiftedPatchTokenization, self).__init__()
        patch_dim = patch_size * patch_size * 5 * channels
        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )
        
    def forward(self, x):
        B, T, C , H, W = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        print(x_with_shifts.shape)
        tokens = self.to_patch_tokens(x_with_shifts)
        return tokens
    
class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        """
        Module for patch embedding.

        Args:
            img_size (int, optional): Input img size. Defaults to 224.
            patch_size (int, optional): Size of patch. Defaults to 16.
            in_channels (int, optional): Number of input channels. Defaults to 3.
            embed_dim (int, optional): Patch embedding dimension. Defaults to 768.
        """
        super(PatchEmbedding, self).__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
    
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, T, C , H, W = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w -> (b t) (h w) c', t=T, b=B)
        return x

class PatchEmbedding3D(nn.Module):
    def __init__(self, patch_size=(2,4,4), in_channels=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dims = embed_dim
        
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
            
    def forward(self, x):
        B, T, C, H, W = x.shape
        
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if T % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - T % self.patch_size[0]))
        
        x = self.proj(x)
        if self.norm is not None:
            T, Wh, Ww = x.shape[2:]
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(B, T, self.embed_dims, Wh, Ww)
        return x
        

class LSA(nn.Module):
    def __init__(self, token_dim: int, head_dims: int, heads: int = 8, dropout: float = 0.) -> None:
        """Constructor for Localized-Self-Attention.
        
        Args:
            token_dim (int): size of token dimension.
            head_dim (int): size of hidden dimension.
            heads (int, optional): Number of heads for layer. Defaults to 8.
        """
        super(LSA, self).__init__()
        inner_dim = head_dims * heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(head_dims ** -0.5)))
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(token_dim, inner_dim * 3, bias=False)
        
        self.unify_heads = nn.Sequential(
            nn.Linear(inner_dim, token_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        queries, keys, values = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        
        dots = einsum(queries, keys, 'b h t1 d, b h t2 d -> b h t1 t2') * self.temperature.exp()
        
        mask = torch.eye(dots.shape[-1], device=dots.device, dtype=torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)
        
        
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        
        out = einsum(attn, values, 'b h t1 t2, b h t2 d -> b h t1 d')
        out = rearrange(out, 'b h t d -> b t (h d)')
        
        out = self.unify_heads(out)
        return out

class MHSA(nn.Module):
    """Class for Multi-Headed Self-Attention.
    """
    def __init__(self, token_dim: int, head_dims: int, heads: int = 8, dropout: float = 0.):
        """Constructor for Multi-Headed Self-Attention.
        
        Args:
            token_dim (int): size of token dimension.
            head_dim (int): size of hidden dimension.
            heads (int, optional): Number of heads for layer. Defaults to 8.
        """
        super().__init__()
        inner_dim = head_dims * heads
        
        self.heads = heads
        self.scale = head_dims ** -0.5
        
        self.to_qkv = nn.Linear(token_dim, inner_dim * 3, bias=False)
        
        self.unify_heads = nn.Sequential(
            nn.Linear(inner_dim, token_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        
        queries, keys, values = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        
        dot = einsum(queries, keys, 'b h t1 d, b h t2 d -> b h t1 t2') * self.scale
        attn = F.softmax(dot, dim=-1)
        
        out = einsum(attn, values, 'b h t1 t2, b h t2 d -> b h t1 d')
        out = rearrange(out, 'b h t d -> b t (h d)')
        
        out = self.unify_heads(out)
        return out
    
class TransformerBlock(nn.Module):
    """Class for Transformer Block.
    """
    def __init__(self, token_dims: int, mlp_dims: int, head_dims: int, heads: int = 8, dropout: float = 0., lsa: bool = False):
        """
        Class for Transformer Block.

        Args:
            token_dims (int): Number of token dimensions.
            mlp_dims (int): Number of hidden dimensions.
            head_dims (int): Number of head dimensions.
            heads (int, optional): Number of heads. Defaults to 8.
            dropout (float, optional): dropout rate. Defaults to 0..
        """
        super().__init__()
        
        self.attention = MHSA(token_dim=token_dims, head_dims=head_dims, heads=heads) if not lsa else LSA(token_dim=token_dims, head_dims=head_dims, heads=heads)

        self.norm1 = nn.LayerNorm(token_dims)
        self.norm2 = nn.LayerNorm(token_dims)
        
        self.mlp = nn.Sequential(
            nn.Linear(token_dims, mlp_dims),
            nn.GELU(),
            nn.Linear(mlp_dims, token_dims),
        )
        
    def forward(self, x):
        out = x + self.attention(self.norm1(x))
        out = x + self.mlp(self.norm2(out))
        return out
    
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio:int = 4, act1=F.silu, act2=torch.sigmoid):
        """
        _summary_

        Args:
            in_channels (int): number of input channels.
            reduction_ratio (int, optional): Reduction ratio for hidden dimension. Defaults to 4.
            act1 (function, optional): First activation. Defaults to F.silu.
            act2 (function, optional): Second Activation. Defaults to torch.sigmoid.
        """
        super(SqueezeExcitation, self).__init__()
        hidden_channels = in_channels // reduction_ratio
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=in_channels, kernel_size=(1, 1), bias=True)
        self.act1 = act1
        self.act2 = act2
    
    def forward(self, x):
        attn = F.adaptive_avg_pool2d(x, (1,1))
        attn = self.conv1(attn)
        attn = self.act1(attn)
        attn = self.conv2(attn)
        attn = self.act2(attn)
        return x * attn
    
class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int, norm_layer: nn.Module, act: nn.Module, conv_layer: nn.Module = nn.Conv2d):
        """
        Constructor for Convolutional Block.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): kernel size.
            stride (int): stride.
            groups (int): number of groups.
            norm_layer (nn.Module): Normalization layer.
            act (nn.Module): Activation layer.
            conv_layer (nn.Module, optional): Type of convolution. Defaults to nn.Conv2d.
        """
        super(ConvBNAct, self).__init__(
            conv_layer(in_channels=in_channels, 
                       out_channels=out_channels, 
                       kernel_size=kernel_size, 
                       stride=stride, 
                       padding= (kernel_size - 1) // 2, 
                       groups=groups, 
                       bias=False),
            norm_layer(out_channels),
            act()
        )
        
class StochasticDepth(nn.Module):
    def __init__(self, prob: float, mode: str):
        """
        Stochastic Depth

        Args:
            prob (float): probability of dying
            mode (str): "row" or "all". If "row", then each row of the tensor has a probability of dying. If "all", then the entire tensor has a probability of dying.
        """
        super(StochasticDepth, self).__init__()
        self.prob = prob
        self.survival_prob = 1 - self.prob
        self.mode = mode
    
    def forward(self, x):
        if self.prob == 0 or not self.training:
            return x
        else:
            shape = [x.shape[0]] + [1] * (x.ndim - 1) if self.mode == "row" else [1]
            return x * torch.empty(shape, device=x.device).bernoulli_(self.survival_prob).div_(self.survival_prob)
    
def adjust_channels(channel, factor, divisible=8):
    new_channel = channel * factor
    divisible_channel = max(divisible, (int(new_channel + divisible / 2) // divisible) * divisible)
    divisible_channel += divisible if divisible_channel < 0.9 * new_channel else 0
    return divisible_channel
    
class MBConv(nn.Module):
    def __init__(self, expand_ratio: int, kernel_size: int, stride: int, in_channels: int, out_channels: int, use_se: bool, fused: bool, act=partial(nn.SiLU, inplace=True), norm_layer=nn.BatchNorm2d, sd_prob: float = 0.):
        """
        Args:
            expand_ratio (int): factor to determine the number of channels in the hidden layer
            kernel_size (int): kernel size of the depthwise convolution
            stride (int): stride of the depthwise convolution
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            use_se (boolean): boolean to determine whether to use squeeze and excitation
            fused (boolean): boolean to determine whether to use fused convolution
            act (nn.Module, optional): activation to be used. Defaults to nn.SiLU.
            norm_layer (Module, optional): type of normalization to be used. Defaults to nn.BatchNorm2d.
            sd_prob (float, optional): stochastic depth probability. Defaults to 0.
        """
        super(MBConv, self).__init__()
        inter_channel = adjust_channels(in_channels, expand_ratio)
        blocks = []
        
        if expand_ratio == 1:
            blocks.append(('fused', ConvBNAct(in_channels=in_channels, 
                                             out_channels=inter_channel, 
                                             kernel_size=kernel_size, 
                                             stride=stride, 
                                             groups=1, 
                                             norm_layer=norm_layer, 
                                             act=act)))
        elif fused:
            blocks.append(('fused', ConvBNAct(in_channels=in_channels, 
                                    out_channels=inter_channel, 
                                    kernel_size=kernel_size, 
                                    stride=stride, 
                                    groups=1, 
                                    norm_layer=norm_layer, 
                                    act=act)))
            blocks.append(('fused_point_wise', ConvBNAct(in_channels=inter_channel,
                                                        out_channels=out_channels,
                                                        kernel_size=1,
                                                        stride=1,
                                                        groups=1,
                                                        norm_layer=norm_layer,
                                                        act=nn.Identity)))
        else:
            blocks.append(('linear_bottle_neck', ConvBNAct(in_channels=in_channels,
                                                          out_channels=inter_channel,
                                                          kernel_size=1,
                                                          stride=1,
                                                          groups=1,
                                                          norm_layer=norm_layer,
                                                          act=act)))
            blocks.append(('depth_wise', ConvBNAct(in_channels=inter_channel,
                                                  out_channels=inter_channel,
                                                  kernel_size=kernel_size,
                                                  stride=stride,
                                                  groups=inter_channel,
                                                  norm_layer=norm_layer,
                                                  act=act)))
            blocks.append(('se', SqueezeExcitation(inter_channel, 4 * expand_ratio)))
            blocks.append(('point_wise', ConvBNAct(in_channels=inter_channel,
                                                  out_channels=out_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  groups=1,
                                                  norm_layer=norm_layer,
                                                  act=nn.Identity)))
        
        self.block = nn.Sequential(OrderedDict(blocks))
        self.use_skip_connection = stride == 1 and in_channels == out_channels
        self.stochastic_depth = StochasticDepth(sd_prob, "row")
    
    def forward(self, x):
        out = self.block(x)
        if self.use_skip_connection:
            out = self.stochastic_depth(out) + x
        return out
    
    
class FeatureExtraction(nn.Module):
    def __init__(self, dim: int, reduce_dim=False) -> None:
        """
        FeatureExtraction Block

        Args:
            dim (_type_): _description_
            reduce_dim (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.conv = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(3, dim, kernel_size=3, stride=2, padding=1, bias=False)),
                ('activation', nn.GELU()),
                ('SqueezeExcitation', SqueezeExcitation(dim)),
                ('conv2', nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False)),
            ])
        )
        
        if reduce_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.reduce_dim = reduce_dim

    def forward(self, x):
        x = x.contiguous()
        x = x + self.conv(x)
        if self.reduce_dim:
            x = self.pool(x)
        return x
    
class GlobalQueryGen(nn.Module):
    def __init__(self, dim: int, input_resolution, image_resolution, window_size, num_heads):
        """
        GlobalQuery Generator Block

        Args:
            dim (int): _description_
            input_resolution (_type_): _description_
            image_resolution (_type_): _description_
            window_size (_type_): _description_
            num_heads (_type_): _description_
        """
        super().__init__()
        if input_resolution == image_resolution // 4:
            self.to_q_global = nn.Sequential(
                FeatureExtraction(dim, reduce_dim=True),
                FeatureExtraction(dim, reduce_dim=True),
                FeatureExtraction(dim, reduce_dim=True)
            )
        elif input_resolution == image_resolution // 8:
            self.to_q_global = nn.Sequential(
                FeatureExtraction(dim, reduce_dim=True),
                FeatureExtraction(dim, reduce_dim=True),
            )
        elif input_resolution == image_resolution // 16:
            if window_size == input_resolution:
                self.to_q_global = nn.Sequential(
                    FeatureExtraction(dim, reduce_dim=False),
                )
            else:
                self.to_q_global = nn.Sequential(
                    FeatureExtraction(dim, reduce_dim=True),
                )
        elif input_resolution == image_resolution // 32:
            self.to_q_global = nn.Sequential(
                FeatureExtraction(dim, reduce_dim=False),
            )
            
        self.resolution = input_resolution
        self.num_heads = num_heads
        self.N = window_size * window_size
        self.dim_head = dim // num_heads
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        
        x = x.permuate(0, 1, 3, 4, 2).contiguous()
        x = x.reshape(B, T, 1, self.N, self.num_heads, self.dim_head).permute(0, 1, 2, 4, 3, 5)
        return x
        
class ReduceSize(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
            nn.GELU(),
            SqueezeExcitation(dim),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        
        if reduce:
            dim_out = dim * 2
        else:
            dim_out = dim
        
        self.reduction = nn.Conv2d(dim, dim_out, kernel_size=3, stride=2, padding=1,bias=False)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim_out)       
        
    def forward(self, x):
        x = x.contiguous().permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = x + self.conv(x)
        x = self.reduction(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        return x

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torchsummary import summary
from collections import OrderedDict
from functools import partial


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
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
            

class MHSA(nn.Module):
    """Class for Multi-Headed Self-Attention.
    """
    def __init__(self, token_dim, head_dims, heads=8, dropout=0):
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
        batch_size, patch_size, dim = x.shape
        heads = self.heads
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        
        queries, keys, values = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=heads), qkv)
        
        
        dot = einsum(queries, keys, 'b h t1 d, b h t2 d -> b h t1 t2') * self.scale
        attn = F.softmax(dot, dim=-1)
        
        out = einsum(attn, values, 'b h t1 t2, b h t2 d -> b h t1 d')
        out = rearrange(out, 'b h t d -> b t (h d)')
        
        out = self.unify_heads(out)
        return out
    
class TransformerBlock(nn.Module):
    """Class for Transformer Block.
    """
    def __init__(self, token_dims, mlp_dims, head_dims, heads=8, dropout=0.):
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
        
        self.attention = MHSA(token_dim=token_dims, head_dims=head_dims, heads=heads)

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
    def __init__(self, in_channels, reduction_ratio=4, act1=F.silu, act2=torch.sigmoid):
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, norm_layer, act, conv_layer=nn.Conv2d):
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
    def __init__(self, prob, mode):
        """
        Stochastic Depth

        Args:
            prob (int): probability of dying
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
    def __init__(self, expand_ratio, kernel_size, stride, in_channels, out_channels, use_se, fused, act=partial(nn.SiLU, inplace=True), norm_layer=nn.BatchNorm2d, sd_prob=0):
        """
        Args:
            expand_ratio (float): factor to determine the number of channels in the hidden layer
            kernel_size (int): kernel size of the depthwise convolution
            stride (int): stride of the depthwise convolution
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            use_se (boolean): boolean to determine whether to use squeeze and excitation
            fused (boolean): boolean to determine whether to use fused convolution
            act (nn.Module, optional): activation to be used. Defaults to nn.SiLU.
            norm_layer (Module, optional): type of normalization to be used. Defaults to nn.BatchNorm2d.
            sd_prob (int, optional): stochastic depth probability. Defaults to 0.
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
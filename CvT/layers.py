import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torchsummary import summary


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
    def __init__(self, token_dims, mlp_dims, head_dims, heads=8, dropout=0):
        """Constructor for Transformer Block.

        Args:
            token_dims (int): Dimension of input tokens
            mlp_dims (int): Dimension of mlp layer
            heads (int): Number of heads for Multi-Headed Self-Attention
        """
        super().__init__()
        
        self.attention = MHSA(token_dim=token_dims, head_dims=head_dims, heads=heads, dropout=dropout)

        self.norm1 = nn.LayerNorm(token_dims)
        self.norm2 = nn.LayerNorm(token_dims)
        
        self.mlp = nn.Sequential(
            nn.Linear(token_dims, mlp_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dims, token_dims),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        out = x + self.attention(self.norm1(x))
        out = x + self.mlp(self.norm2(out))
        return out
        
class SqueezeExcitation(nn.Module):
    def __init__(self, in_dims, reduced_dims):
        """
        Constructor for Squeeze Excitation Layer.

        Args:
            in_dims (int): Number of input channels
            reduced_dims (int): Number of hidden channels
        """        
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dims, reduced_dims, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(),
            nn.Conv2d(reduced_dims, in_dims, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attn = self.se(x)
        return x * attn
    
class CNNBlock(nn.Module):
    def __init__(self, in_dims, out_dims, kernel_size, stride, padding, groups=1):
        """
        Constructor for CNN Block.

        Args:
            in_dims (int): Number of input channels
            out_dims (int): Number of output channels
            kernel_size (int): kernel size
            stride (int): stride
            padding (int): padding
            groups (int, optional): _description_. Defaults to 1.
        """
        super(CNNBlock, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_dims),
            nn.SiLU()
        )
    
    def forward(self, x):
        return self.cnn(x)

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_dims, out_dims, kernel_size, stride, padding, expand_ratio, reduction=4, survival_prob=0.8):
        """
        Constructor for Inverted Residual Block.

        Args:
            in_dims (int): Number of input channels
            out_dims (int): Number of output channels
            kernel_size (int): kernel size
            stride (int): stride
            padding (int): padding
            expand_ratio (int): Expansion ratio
            reduction (int, optional): Reduction ratio. Defaults to 4.
            survival_prob (float, optional): Stochastic depth probability. Defaults to 0.8.
        """
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_dims == out_dims and stride == 1
        self.hidden_dims = in_dims * expand_ratio
        self.expand = in_dims != self.hidden_dims
        self.reduced_dims = int(in_dims / reduction) 
        if self.reduced_dims < 1:
            self.reduced_dims = 1
        
        if self.expand:
            self.expand_conv = CNNBlock(in_dims, self.hidden_dims, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Sequential(
            CNNBlock(self.hidden_dims, self.hidden_dims, kernel_size=kernel_size, stride=stride, padding=padding, groups=self.hidden_dims),
            SqueezeExcitation(self.hidden_dims, self.reduced_dims),
            nn.Conv2d(self.hidden_dims, out_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dims)
        )
    
    def stochastic_depth(self, x):
        if not self.training:
            return x
        
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        
        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)

if __name__ == '__main__':
    inp = torch.randn(32, 3, 224, 224)
    
    mb = InvertedResidualBlock(in_dims=3, out_dims=24, kernel_size=3, stride=1, padding=1, expand_ratio=4)
    tf = TransformerBlock(token_dims=224*224, mlp_dims=16, head_dims=16, heads=8)
    
    x = mb(inp)
    print(f"Shape of x after MBConvBlock: {x.shape}")
    x = x.view(x.shape[0], x.shape[1], -1)
    x = tf(x)
    print(f"Shape of x after TransformerBlock: {x.shape}")

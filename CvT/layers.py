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
        
class SqeezeExcitation(nn.Module):
    def __init__(self, in_dims, out_dims, reduction=4):
        super(SqeezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_dims, in_dims // reduction),
            nn.SiLU(),
            nn.Linear(in_dims // reduction, out_dims),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
class MBConvBlock(nn.Module):
    def __init__(self, in_dims, out_dims, stride, expansion_factor, se, reduction=4):
        super(MBConvBlock, self).__init__()
        assert stride in [1, 2]
        
        self.identity = (in_dims == out_dims) and (stride == 1)
        
        hidden_dim = round(in_dims * expansion_factor)
        if se:
            self.conv = nn.Sequential(
                nn.Conv2d(in_dims, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                SqeezeExcitation(in_dims, hidden_dim, reduction=reduction),
                
                nn.Conv2d(hidden_dim, out_dims, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_dims),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_dims, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                
                nn.Conv2d(hidden_dim, out_dims, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_dims),
            )
    
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)
            


if __name__ == '__main__':
    inp = torch.randn(32, 3, 224, 224)
    
    mb = MBConvBlock(in_dims=3, out_dims=16, stride=1, expansion_factor=1, se=True)
    tf = TransformerBlock(token_dims=224*224, mlp_dims=16, head_dims=16, heads=8)
    
    x = mb(inp)
    print(f"Shape of x after MBConvBlock: {x.shape}")
    x = x.view(x.shape[0], x.shape[1], -1)
    x = tf(x)
    print(f"Shape of x after TransformerBlock: {x.shape}")
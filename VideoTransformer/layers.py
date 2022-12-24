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
        self.token_dim = token_dim
        self.head_dim = head_dims
        self.heads = heads
        self.scale = head_dims ** -0.5
        
        self.to_keys = nn.Linear(token_dim, head_dims * heads, bias=False)
        self.to_queries = nn.Linear(token_dim, head_dims * heads, bias=False)
        self.to_values = nn.Linear(token_dim, head_dims * heads, bias=False)

        self.unify_heads = nn.Linear(heads * head_dims, token_dim)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        batch_size, patch_size, dim = x.shape
        heads = self.heads
        
        queries = self.to_queries(x).view(batch_size, heads, patch_size, self.head_dim)
        keys = self.to_keys(x).view(batch_size, heads, patch_size, self.head_dim)
        values = self.to_values(x).view(batch_size, heads, patch_size, self.head_dim)
        
        dot = einsum(queries, keys, 'b h t1 d, b h t2 d -> b h t1 t2') * self.scale
        attn = F.softmax(dot, dim=-1)
        
        out = einsum(attn, values, 'b h t1 t2, b h t2 d -> b h t1 d')
        out = rearrange(out, 'b h t d -> b t (h d)')
        
        out = self.unify_heads(out)
        out = self.dropout(out)
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
        
if __name__ == '__main__':
    model = TransformerBlock(128, 128, 32)
    print(summary(model, (32, 128), device='cpu'))
    test = torch.randn(2, 32, 128)
    result = model(test)
    print("Shape after Transformer block",result.shape)

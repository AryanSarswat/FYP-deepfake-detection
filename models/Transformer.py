import torch
import torch.nn as nn
from layers import TransformerBlock
from util import trunc_normal_

class Transformer(nn.Module):
    """Class for Transformer.
    """
    def __init__(self, token_dim: int, depth: int, head_dims: int, heads: int, mlp_dim: int, dropout:float = 0, lsa:bool = False):
        """Constructor for Transformer.

        Args:
            token_dim (int): Dimension of input tokens
            depth (int): Number of Transformer Blocks
            head_dims (int): dimension of each head
            heads (int): Number of heads for layer
            mlp_dim (int): Dimension of MLP
            dropout (int, optional): Dropout probability. Defaults to 0.
        """
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)
        
        for _ in range(depth):
            self.layers.append(TransformerBlock(token_dims=token_dim, mlp_dims=mlp_dim, head_dims=head_dims, heads=heads, dropout=dropout, lsa=lsa))
            
        self.apply(self._init_weights)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

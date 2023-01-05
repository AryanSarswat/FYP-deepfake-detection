from layers import TransformerBlock
import torch
import torch.nn as nn

class Transformer(nn.Module):
    """Class for Transformer.
    """
    def __init__(self, token_dim: int, depth: int, head_dims: int, heads: int, mlp_dim: int, dropout:float = 0):
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
            self.layers.append(TransformerBlock(token_dims=token_dim, mlp_dims=mlp_dim, head_dims=head_dims, heads=heads))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x
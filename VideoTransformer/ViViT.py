import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat, einsum
from einops.layers.torch import Rearrange

from layers import TransformerBlock

class Transformer(nn.Module):
    """Class for Transformer.
    """
    def __init__(self, token_dim, depth, head_dims, heads, mlp_dim, dropout=0):
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
        
        for _ in range(depth):
            self.layers.append(TransformerBlock(token_dims=token_dim, mlp_dims=mlp_dim, head_dims=head_dims, heads=heads, dropout=dropout))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class ViViT(nn.Module):
    def __init__(self, num_frames, patch_size, in_channels, height, width, dim=192, depth=4, heads=3, head_dims=64, dropout=0):
        super().__init__()

if __name__ == '__main__':
    test = torch.randn(3, 120, 256).cuda()
    model = Transformer(token_dim=256, depth=4, head_dims=64, heads=3, mlp_dim=512).cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters {pytorch_total_params:,}")
    result = model(test)
    print(result.shape)
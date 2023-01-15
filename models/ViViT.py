from collections import OrderedDict

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
from .layers import PatchEmbedding, ShiftedPatchTokenization
from torch import nn
from torchsummary import summary
from .Transformer import Transformer
from .util import trunc_normal_


class ViViT(nn.Module):
    """Class for Video Vision Transformer.
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
        
        self.to_patch_embedding = PatchEmbedding(img_size=height, patch_size=patch_size, in_channels=in_channels, embed_dim=dim) if not spt else ShiftedPatchTokenization(dim=dim, patch_size=patch_size, channels=in_channels)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.spatial_token = nn.Parameter(torch.randn(1, 1, dim))
        self.spatial_transformer = Transformer(token_dim=dim, depth=depth, head_dims=head_dims, heads=heads, mlp_dim=dim*scale_dim, dropout=dropout, lsa=lsa)
        
        self.temporal_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
        self.temporal_transformer = Transformer(token_dim=dim, depth=depth, head_dims=head_dims, heads=heads, mlp_dim=dim*scale_dim, dropout=dropout, lsa=lsa)
        
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
        
    def forward(self, x):
        b, t, c, h, w = x.shape
        
        x = self.to_patch_embedding(x)
        
        _ , n, d = x.shape
        
        cls_spatial_token = repeat(self.spatial_token, '() n d -> (b t) n d', b=b, t=t)
        x = torch.cat((cls_spatial_token, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.spatial_transformer(x)
        
        # Taking only the spatial token
        x = x[:,0]
        
        # Unfold the batch and time dimensions
        x = rearrange(x, '(b t) ... -> b t ...', b=b, t=t)
        
        x += self.temporal_embedding
        x = self.dropout(x)
        
        x = self.temporal_transformer(x)
        
        x = x[:,0]
        
        return self.head(x)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
def create_model(num_frames: int, patch_size: int, in_channels: int, height: int, width: int, dim: int = 192, depth: int = 4, heads: int = 3, head_dims: int = 64, dropout: float = 0, scale_dim: int = 4):
    return ViViT(num_frames, patch_size, in_channels, height, width, dim, depth, heads, head_dims, dropout, scale_dim)

def load_model(path: str):
    model = create_model(num_frames=16, patch_size=16, in_channels=3, height=256, width=256, dim=192, depth=6, heads=6, head_dims=64, dropout=0, scale_dim=4)
    model = model.load_state_dict(torch.load(path))
    return model
        
if __name__ == '__main__':
    HEIGHT = 256
    WIDTH = 256
    NUM_FRAMES = 16
    PATCH_SIZE = 16
    
    test = torch.randn(2, NUM_FRAMES, 3, HEIGHT, WIDTH).cuda()
    model = ViViT(num_frames=NUM_FRAMES, patch_size=PATCH_SIZE, in_channels=3, height=HEIGHT, width=WIDTH, dim=256, depth=8, heads=6, head_dims=128, spt=True, lsa=True).cuda()
    print(summary(model, (NUM_FRAMES, 3, HEIGHT, WIDTH)))
    result = model(test)
    print(result.shape)
    print(result)

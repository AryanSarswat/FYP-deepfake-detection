import torch
import torch.nn.functional as F
from torchsummary import summary
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
from collections import OrderedDict
from Transformer import Transformer
from torch import nn


class ViViT(nn.Module):
    """Class for Video Vision Transformer.
    """
    def __init__(self, num_frames, patch_size, in_channels, height, width, dim=192, depth=4, heads=3, head_dims=64, dropout=0, scale_dim=4):
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
        patch_dim = in_channels * patch_size * patch_size
        
        self.to_patch_embedding = nn.Sequential(OrderedDict([
            ('Rearrange', Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)),
            ('ToPatch', nn.Linear(patch_dim, dim)),
        ]))
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.spatial_token = nn.Parameter(torch.randn(1, 1, dim))
        self.spatial_transformer = Transformer(token_dim=dim, depth=depth, head_dims=head_dims, heads=heads, mlp_dim=dim*scale_dim, dropout=dropout)
        
        self.temporal_embedding = nn.Parameter(torch.randn(1, num_frames + 1, dim))
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(token_dim=dim, depth=depth, head_dims=head_dims, heads=heads, mlp_dim=dim*scale_dim, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.to_patch_embedding(x)
        
        b, t, n, _ = x.shape
        
        cls_spatial_tokens = repeat(self.spatial_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((cls_spatial_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)
        
        # Fold the batch and time dimensions
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.spatial_transformer(x)
        
        # Unfold the batch and time dimensions
        x = rearrange(x[:,0], '(b t) ... -> b t ...', b=b)
        
        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x += self.temporal_embedding[:, :(t + 1)]
        x = self.dropout(x)
        
        x = self.temporal_transformer(x)
        
        x = x[:,0]
        
        return self.classifier(x)
        
def create_model(num_frames, patch_size, in_channels, height, width, dim=192, depth=4, heads=3, head_dims=64, dropout=0, scale_dim=4):
    return ViViT(num_frames, patch_size, in_channels, height, width, dim, depth, heads, head_dims, dropout, scale_dim)
        
        
if __name__ == '__main__':
    HEIGHT = 256
    WIDTH = 256
    NUM_FRAMES = 32
    PATCH_SIZE = 16
    
    test = torch.randn(2, NUM_FRAMES, 3, HEIGHT, WIDTH).cuda()
    model = ViViT(num_frames=NUM_FRAMES, patch_size=PATCH_SIZE, in_channels=3, height=HEIGHT, width=WIDTH, dim=256, depth=8, heads=6, head_dims=128).cuda()
    print(summary(model, (NUM_FRAMES, 3, HEIGHT, WIDTH)))
    result = model(test)
    print(result.shape)
    print(result)

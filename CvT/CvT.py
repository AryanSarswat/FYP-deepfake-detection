import torch
import torch.nn.functional as F
from torchsummary import summary
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
from layers import TransformerBlock
from torch import nn
from math import ceil
from torchvision.models import efficientnet_v2_s

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

class ConvolutionalVisionTransformer(nn.Module):
    """Class for Video Vision Transformer.
    """
    def __init__(self, num_frames, t_dim=192, t_depth=4, t_heads=3, t_head_dims=64, dropout=0, scale_dim=4):
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
            dropout (float, optional): _description_. Defaults to 0.
            scale_dim (int, optional): _description_. Defaults to 4.
        """
        super(ConvolutionalVisionTransformer, self).__init__()
        
        # EfficientNet style Convolution to extract features from frames
        self.efficientnet_backbone = efficientnet_v2_s()
        self.efficientnet_backbone.classifier = nn.Linear(1280, t_dim)
    
        # Transformer for temporal dimension
        self.temporal_embedding = nn.Parameter(torch.randn(1, num_frames + 1, t_dim))
        self.temporal_token = nn.Parameter(torch.randn(1, 1, t_dim))
        self.temporal_transformer = Transformer(token_dim=t_dim, depth=t_depth, head_dims=t_head_dims, heads=t_heads, mlp_dim=t_dim*scale_dim, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(t_dim),
            nn.Linear(t_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, t, c, h, w = x.shape
        
        # Fold time into batch dimension
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        
        x = self.efficientnet_backbone(x)
        
        # Unfold time from batch dimension
        x = rearrange(x, '(b t) d -> b t d', b=b, t=t)
        
        # Add temporal token
        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x += self.temporal_embedding[:, :(t + 1)]
        x = self.temporal_transformer(x)
        
        # Take only the cls_temporal_token
        x = x[:,0]
        
        return self.classifier(x)
        
def create_model(num_frames, dim=192, depth=4, heads=3, head_dims=64, dropout=0, scale_dim=4):
    return ConvolutionalVisionTransformer(num_frames=num_frames, 
                                          t_dim=dim, t_depth=depth, t_heads=heads, 
                                          t_head_dims=head_dims, dropout=dropout, scale_dim=scale_dim)
        
        
if __name__ == '__main__':
    HEIGHT = 256
    WIDTH = 256
    NUM_FRAMES = 32
    
    test = torch.randn(2, NUM_FRAMES, 3, HEIGHT, WIDTH)
    model = create_model(num_frames=NUM_FRAMES)
    result = model(test)
    print(f"Shape of output : {result.shape}")
    print(f"Number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(summary(model, (NUM_FRAMES, 3, HEIGHT, WIDTH), device='cuda'))

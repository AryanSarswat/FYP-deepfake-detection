import torch
import torch.nn.functional as F
from torchsummary import summary
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
from EfficientNetV2 import create_efficientnetv2_backbone
from Transformer import Transformer
from torch import nn
from math import ceil

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
        self.efficientnet_backbone = create_efficientnetv2_backbone()
        
        self.fc = nn.Linear(1280, t_dim)
    
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
        
        # Map to token dimension
        x = self.fc(x)
        
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
    NUM_FRAMES = 8
    
    
    test = torch.randn(1, NUM_FRAMES, 3, HEIGHT, WIDTH).cuda()
    model = create_model(num_frames=NUM_FRAMES).cuda()
    output = model(test)
    print(output.shape)
    print(summary(model, (NUM_FRAMES, 3, HEIGHT, WIDTH)))
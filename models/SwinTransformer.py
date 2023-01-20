from math import ceil
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
from layers import TransformerBlock, ConvBNAct, MBConv, ShiftedPatchTokenization
from torch import nn
from torchsummary import summary
from Transformer import Transformer


class MiniConvolutionalVisionTransformer(nn.Module):
    """Class for Video Vision Transformer.
    """
    def __init__(self, num_frames, patch_size=16, in_channels=3, t_dim=192, t_depth=4, t_heads=3, t_head_dims=64, dropout=0., scale_dim=4, height=224, width=224):
        """Constructor for ViViT.

        Args:
            num_frames (int): Number of frames in video
            patch_size (int): Size of patch
            in_channels (int): Number of channels in input
            height (int): Height of input
            width (int): Width of input
            num_classes (int): Number of classes
            dim (int, optional): Token dimensions. Defaults to 192.
            depth (int, optional): Number of transformer blocks. Defaults to 4.
            heads (int, optional): Number of attention heads. Defaults to 3.
            head_dims (int, optional): Number of dimension of attention head. Defaults to 64.
            dropout (float, optional): Dropout rate. Defaults to 0.
            scale_dim (int, optional): Scale factor for MLP. Defaults to 4.
        """
        super(MiniConvolutionalVisionTransformer, self).__init__()
        
        # Convolutions to extract features
        self.conv_backbone = nn.Sequential(OrderedDict([
            ('stem', ConvBNAct(in_channels=in_channels, out_channels=24, kernel_size=1, stride=1, groups=1, norm_layer=nn.BatchNorm2d, act=nn.SiLU)),
            ('conv1', ConvBNAct(in_channels=24, out_channels=48, kernel_size=3, stride=1, groups=1, norm_layer=nn.BatchNorm2d, act=nn.SiLU)),
            ('conv2', ConvBNAct(in_channels=48, out_channels=64, kernel_size=3, stride=1, groups=1, norm_layer=nn.BatchNorm2d, act=nn.SiLU)),
            ('conv3', ConvBNAct(in_channels=64, out_channels=128, kernel_size=5, stride=1, groups=1, norm_layer=nn.BatchNorm2d, act=nn.SiLU)),
        ]))
                                           
        num_patches = (height // patch_size) * (width // patch_size)
        
        self.to_patch_embedding = ShiftedPatchTokenization(dim=t_dim, patch_size=patch_size, channels=128)
        
        # Transformer for spatial dimension
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, t_dim))
        self.spatial_token = nn.Parameter(torch.randn(1, 1, t_dim))
        self.spatial_transformer = Transformer(token_dim=t_dim, depth=t_depth, head_dims=t_head_dims, heads=t_heads, mlp_dim=t_dim*scale_dim, dropout=dropout, lsa=True)
        
        # Transformer for temporal dimension
        self.temporal_embedding = nn.Parameter(torch.randn(1, num_frames + 1, t_dim))
        self.temporal_token = nn.Parameter(torch.randn(1, 1, t_dim))
        self.temporal_transformer = Transformer(token_dim=t_dim, depth=t_depth, head_dims=t_head_dims, heads=t_heads, mlp_dim=t_dim*scale_dim, dropout=dropout, lsa=True)
        
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
        
        x = self.conv_backbone(x)
        
        x = rearrange(x, '(b t) c h w -> b t c h w', b=b, t=t)
        
        x = self.to_patch_embedding(x)
        
        _, n , d = x.shape

        # Unfold time from batch dimension
        cls_spatial_token = repeat(self.spatial_token, '() n d -> (b t) n d', b=b, t=t)
        x = torch.cat((cls_spatial_token, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.spatial_transformer(x)
        
        x = x[:,0]
        
        x = rearrange(x, '(b t) ... -> b t ...', b=b, t=t)
        
        # Add temporal token
        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x += self.temporal_embedding[:, :(t + 1)]
        x = self.dropout(x)
        x = self.temporal_transformer(x)
        
        # Take only the cls_temporal_token
        x = x[:,0]
        
        return self.classifier(x)
        
def create_model(num_frames, in_channels, dim=192, depth=4, heads=3, head_dims=64, dropout=0., scale_dim=4, height=224, weight=224, patch_size=16):
    return MiniConvolutionalVisionTransformer(num_frames=num_frames, in_channels=in_channels,
                                          t_dim=dim, t_depth=depth, t_heads=heads, 
                                          t_head_dims=head_dims, dropout=dropout, scale_dim=scale_dim, height=height, width=weight, patch_size=patch_size)
    
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model
        
        
if __name__ == '__main__':
    HEIGHT = 224
    WIDTH = 224
    NUM_FRAMES = 16
    
    test = torch.randn(1, NUM_FRAMES, 3, HEIGHT, WIDTH).cuda()
    model = create_model(num_frames=NUM_FRAMES).cuda()
    result = model(test)
    print(f"Shape of output : {result.shape}")
    print(f"Number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(summary(model, (NUM_FRAMES, 3, HEIGHT, WIDTH), device='cuda'))
    

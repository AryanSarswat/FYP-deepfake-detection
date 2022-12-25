import torch
import torch.nn.functional as F
from torchsummary import summary
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
from layers import TransformerBlock, MBConvBlock
from torch import nn


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
    def __init__(self, num_frames, in_channels, conv_config, width_multiplier, dim=192, depth=4, heads=3, head_dims=64, dropout=0, scale_dim=4):
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
        super(ConvolutionalVisionTransformer, self).__init__()
        
        # EfficientNet style Convolution to extract features from frames
        
        self.features = [nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )]
        
        for expansion_factor, channels, num_layers, stride, se in conv_config:
            out_channels = int(channels * width_multiplier)
            for i in range(num_layers):
                self.features.append(MBConvBlock(in_dims=in_channels, out_dims=out_channels, stride=stride if 1 == 0 else 1, expansion_factor=expansion_factor, se=se))
                in_channels = out_channels
                
        self.features = nn.Sequential(*self.features)
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Transformer for temporal dimension
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
        b, t, c, h, w = x.shape
        
        # Fold time into batch dimension
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        
        x = self.features(x)
        x = self.conv_out(x)
        
        # Unfold time from batch dimension
        x = rearrange(x, '(b t) c h w -> b t (c h w)', b=b, t=t)
        
        # Add temporal token
        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x += self.temporal_embedding[:, :(t + 1)]
        x = self.temporal_transformer(x)
        
        # Take only the cls_temporal_token
        x = x[:,0]
        
        return self.classifier(x)
        
def create_model(num_frames, in_channels, conv_config, width_multiplier, dim=192, depth=4, heads=3, head_dims=64, dropout=0, scale_dim=4):
    return ConvolutionalVisionTransformer(num_frames=num_frames, 
                                          in_channels=in_channels, 
                                          conv_config=conv_config, 
                                          width_multiplier=width_multiplier, 
                                          dim=dim, depth=depth, heads=heads, 
                                          head_dims=head_dims, dropout=dropout, scale_dim=scale_dim)
        
        
if __name__ == '__main__':
    HEIGHT = 256
    WIDTH = 256
    NUM_FRAMES = 32
    
    effnetv2_s = [
        [1,  24,  1, 1, 0],
        [1,  48,  1, 2, 0],
        [1,  64,  1, 2, 0],
        [1, 128,  2, 2, 1],
        [1, 160,  2, 1, 1],
        [1, 256,  2, 2, 1],
    ]
    
    test = torch.randn(1, NUM_FRAMES, 3, HEIGHT, WIDTH)
    
    model = create_model(num_frames=NUM_FRAMES, in_channels=3, conv_config=effnetv2_s, width_multiplier=1.0)
    result = model(test)
    print(f"Shape of output : {result.shape}")
    print(f"Number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(model)

import torch
from einops import rearrange, repeat
from torch import nn
from torchsummary import summary

import torchvision
from .Transformer import Transformer


class ConvolutionalVisionTransformer(nn.Module):
    """Class for Video Vision Transformer.
    """
    def __init__(self, num_frames, t_dim=192, t_depth=4, t_heads=3, t_head_dims=64, dropout=0., scale_dim=4, lsa=False, in_channels=3):
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
        super(ConvolutionalVisionTransformer, self).__init__()
        
        # EfficientNet style Convolution to extract features from frames
        self.efficientnet_backbone = torchvision.models.efficientnet_v2_m(weights="IMAGENET1K_V1")
        self.efficientnet_backbone.classifier = nn.Identity()
        
        if in_channels != 3:
            self.efficientnet_backbone.features[0] = nn.Sequential(
                nn.Conv2d(in_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                nn.SiLU(inplace=True),                
            )
        
        self.fc = nn.Linear(1280, t_dim)
    
        # Transformer for temporal dimension
        self.temporal_embedding = nn.Parameter(torch.randn(1, num_frames + 1, t_dim))
        self.temporal_token = nn.Parameter(torch.randn(1, 1, t_dim))
        self.temporal_transformer = Transformer(token_dim=t_dim, depth=t_depth, head_dims=t_head_dims, heads=t_heads, mlp_dim=t_dim*scale_dim, dropout=dropout, lsa=lsa)
        
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(t_dim)
        
        self.spatial_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )
        
        self.temporal_head = nn.Sequential(
            nn.Linear(t_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(t_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, t, c, h, w = x.shape
        
        # Fold time into batch dimension
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        
        x = self.efficientnet_backbone(x)
        
        
        spatial_classification = self.spatial_head(x)
        spatial_classification = spatial_classification.view(b, t)
        
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
        x = self.norm(torch.mean(x, dim=1))
        
        vectors = x
        
        temporal_classification = self.temporal_head(x)
        
        return temporal_classification, spatial_classification, vectors
        
def create_model(num_frames, dim=192, depth=4, heads=3, head_dims=64, dropout=0., scale_dim=4, lsa=False, in_channels=3):
    return ConvolutionalVisionTransformer(num_frames=num_frames, 
                                          t_dim=dim, t_depth=depth, t_heads=heads, 
                                          t_head_dims=head_dims, dropout=dropout, scale_dim=scale_dim, lsa=lsa, in_channels=in_channels)
    
def load_model(base_model, weights_path):
    weights = torch.load(weights_path)
    base_model.load_state_dict(weights)

    base_model.eval()
    return base_model
        
        
if __name__ == '__main__':
    HEIGHT = 224
    WIDTH = 224
    NUM_FRAMES = 16
    
    test = torch.randn(2, NUM_FRAMES, 3, HEIGHT, WIDTH).cuda()
    model = create_model(num_frames=NUM_FRAMES).cuda()
    temporal, spatial, vectors = model(test)
    print(temporal.shape)
    print(spatial.shape)
    print(vectors.shape)
    print(f"Number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(summary(model, (NUM_FRAMES, 3, HEIGHT, WIDTH), device='cuda'))
    

import torch
import torch.nn as nn
import torch.functional as F
from einops import rearrange, repeat
from .Transformer import Transformer
from .util import trunc_normal_
import timm

class PretrainedGCViViT(nn.Module):
    def __init__(self, dim, depth, head_dims, heads, scale_dim, dropout, lsa, num_frames, in_channels):
        super().__init__()
        
        self.spatial_transformer = timm.create_model('gcvit_tiny', num_classes=0, pretrained=True)
        
        if in_channels != 3:
            weights = self.spatial_transformer.stem.conv1.weight
            self.spatial_transformer.stem.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1))
            self.spatial_transformer.stem.conv1.weight = nn.Parameter(torch.stack([torch.mean(weights, 1)]*in_channels, dim=1))            
        
        self.temporal_embedding = nn.Parameter(torch.randn(1, num_frames, 512))
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(token_dim=dim, depth=depth, head_dims=head_dims, heads=heads, mlp_dim=dim*scale_dim, dropout=dropout, lsa=lsa)
        
        self.dropout = nn.Dropout(dropout)
        
        self.spatial_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )
        
        self.temporal_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )
        
        # Initialize weights
        trunc_normal_(self.temporal_embedding, std=.02)
        
        # Freeze Backbone
        self.freeze_weights(self.spatial_transformer)
        
    def forward(self, x):
        
        B, F, C, H, W = x.shape
        
        # Process spatial dimensions first
        x = x.reshape(B*F, C, H, W)
        x = self.spatial_transformer(x)
        x = self.dropout(x)
        
        spatial_features = x.reshape(B * F, -1)
        spatial_classification = self.spatial_head(spatial_features)
        spatial_classification = spatial_classification.reshape(B, F)
        
        # Unfold time
        x = x.reshape(B, F, -1)
        cls_token = repeat(self.temporal_cls_token, '() n d -> b n d', b=B)
        #x = torch.cat((cls_token, x), dim=1)
        x = x + self.temporal_embedding
        x = self.temporal_transformer(x)
        x = torch.mean(x[:, 1:], dim=1)
        x = self.dropout(x)
        
        classification_vectors = x
        
        temporal_classification = self.temporal_head(x)
        return temporal_classification, spatial_classification, classification_vectors
    
    def freeze_weights(self, model):
        for param in model.parameters():
            param.requires_grad = False
            
    def unfreeze_weights(self, model):
        for param in model.parameters():
            param.requires_grad = True
            
def create_model(num_frames, in_channels, dim, depth, head_dims, heads, dropout, lsa=False):
    model = PretrainedGCViViT(dim=dim, depth=depth, head_dims=head_dims, heads=heads, scale_dim=4, dropout=dropout, lsa=lsa, num_frames=num_frames, in_channels=in_channels)
    return model
            
if __name__ == "__main__":
    model = PretrainedGCViViT(dim=512, depth=8, head_dims=64, heads=16, scale_dim=4, dropout=0.1, lsa=True, num_frames=16, in_channels=6)
    x = torch.randn(2, 16, 6, 224, 224)
    y, y_s, _ = model(x)
    print(y.shape)
    print(y_s.shape)

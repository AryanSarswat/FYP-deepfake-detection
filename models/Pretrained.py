import torch
import torch.nn as nn
import torch.functional as F
from einops import rearrange, repeat
from .Transformer import Transformer
from .util import trunc_normal_
import timm

class PretrainedGCViViT(nn.Module):
    def __init__(self, dim, depth, head_dims, heads, scale_dim, dropout, lsa, num_frames):
        super().__init__()
        
        self.spatial_transformer = timm.create_model('gcvit_tiny', pretrained=True)
        
        self.temporal_embedding = nn.Parameter(torch.randn(1, num_frames + 1, dim))
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(token_dim=dim, depth=depth, head_dims=head_dims, heads=heads, mlp_dim=dim*scale_dim, dropout=dropout, lsa=lsa)
        
        self.dropout = nn.Dropout(dropout)
        
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
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
        
        # Unfold time
        x = x.reshape(B, F, -1)
        cls_token = repeat(self.temporal_cls_token, '() n d -> b n d', b=B)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.temporal_embedding
        x = self.temporal_transformer(x)
        # Taking only the cls token
        x = x[:, 0]
        x = self.dropout(x)
        
        vectors = x
        
        x = self.head(x)
        return x, vectors
    
    def freeze_weights(self, model):
        for param in model.parameters():
            param.requires_grad = False
            
    def unfreeze_weights(self, model):
        for param in model.parameters():
            param.requires_grad = True
            
def create_model(num_frames, in_channels, dim, depth, head_dims, heads, dropout, lsa=False):
    model = PretrainedGCViViT(dim=dim, depth=depth, head_dims=head_dims, heads=heads, scale_dim=4, dropout=dropout, lsa=lsa, num_frames=num_frames)
    return model
            
if __name__ == "__main__":
    model = PretrainedGCViViT(dim=512, depth=6, head_dims=64, heads=8, scale_dim=4, dropout=0.1, lsa=True, num_frames=16)
    x = torch.randn(2, 16, 3, 224, 224)
    y, _ = model(x)
    print(y.shape)
import torch
import torch.nn.functional as F
from torchsummary import summary
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
from layers import TransformerBlock, SqueezeExcitation, ConvBNAct, MBConv, StochasticDepth
from collections import OrderedDict
from torch import nn
from math import ceil

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
    
class EfficientNetV2(nn.Module):
    def __init__(self, layer_info, out_channels=1280, n_class=0, dropout=0.2, stochastic_depth=0, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d):
        super(EfficientNetV2, self).__init__()
        self.layer_info = layer_info
        self.norm_layer = norm_layer
        self.act = act_layer
        
        self.in_channel = self.layer_info[0][3]
        self.final_stage_channel = self.layer_info[-1][4]
        self.out_channels = out_channels
        
        self.cur_block = 0
        self.num_block = sum([layer[5] for layer in self.layer_info])
        self.stochastic_depth = stochastic_depth
        
        self.stem = ConvBNAct(in_channels=3, 
                              out_channels=self.in_channel, 
                              kernel_size=3, 
                              stride=2, 
                              groups=1, 
                              norm_layer=norm_layer, 
                              act=act_layer)
        self.blocks = self._make_blocks(layer_info)
        self.head = nn.Sequential(OrderedDict([
            ('bottle_neck', ConvBNAct(in_channels=self.final_stage_channel, 
                                      out_channels=out_channels, 
                                      kernel_size=1, 
                                      stride=1, 
                                      groups=1, 
                                      norm_layer=norm_layer, 
                                      act=act_layer)),
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', nn.Flatten()),
            ('dropout', nn.Dropout(p=dropout, inplace=True)),
            ('classifier', nn.Linear(out_channels, n_class) if n_class > 0 else nn.Identity()),
        ]))
        
        self._init_weights()
        
    def _make_blocks(self, layer_info):
        layers = []
        for i in range(len(layer_info)):
            expand_ratio, kernel_size, stride, in_channels, out_channels, num_layers, use_se, fused = layer_info[i]
            for j in range(num_layers):
                layers.append((f'MBConv{i}_{j}', MBConv(expand_ratio = expand_ratio, 
                                    kernel_size = kernel_size, 
                                    stride = stride, 
                                    in_channels = in_channels, 
                                    out_channels = out_channels,
                                    use_se = use_se,
                                    fused = fused,
                            )))
                in_channels = out_channels
                stride = 1
        return nn.Sequential(OrderedDict(layers))
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))

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
        EFFICIENTNETV2_S_CONFIG = [
            # expand_ratio, kernel_size, stride, in_channels, out_channels, layers, use_se, fused
            (1, 3, 1, 24, 24, 2, False, True),
            (4, 3, 2, 24, 48, 4, False, True),
            (4, 3, 2, 48, 64, 4, False, True),
            (4, 3, 2, 64, 128, 6, True, False),
            (6, 3, 1, 128, 160, 9, True, False),
            (6, 3, 2, 160, 256, 15, True, False),
        ]
        
        self.efficientnet_backbone = EfficientNetV2(layer_info=EFFICIENTNETV2_S_CONFIG)
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
    
    model = EfficientNetV2([
            # expand_ratio, kernel_size, stride, in_channels, out_channels, layers, use_se, fused
            (1, 3, 1, 24, 24, 2, False, True),
            (4, 3, 2, 24, 48, 4, False, True),
            (4, 3, 2, 48, 64, 4, False, True),
            (4, 3, 2, 64, 128, 6, True, False),
            (6, 3, 1, 128, 160, 9, True, False),
            (6, 3, 2, 160, 256, 15, True, False),
        ]).cuda()
    
    print(summary(model, (3, 256, 256)))
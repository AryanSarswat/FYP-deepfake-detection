import typing
from collections import OrderedDict

import torch
import torch.nn as nn

from layers import ConvBNAct, MBConv


class EfficientNetV2(nn.Module):
    def __init__(self, layer_info: typing.List[tuple], out_channels: int = 1280, num_classes: int = 0, dropout: float = 0.2, stochastic_depth: int = 0, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d, in_channels=3):
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
        
        self.stem = ConvBNAct(in_channels=in_channels, 
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
            ('classifier', nn.Linear(out_channels, num_classes) if num_classes > 0 else nn.Identity()),
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


EFFICIENTNETV2_S_CONFIG = [
    # expand_ratio, kernel_size, stride, in_channels, out_channels, layers, use_se, fused
    (1, 3, 1, 24, 24, 2, False, True),
    (4, 3, 2, 24, 48, 4, False, True),
    (4, 3, 2, 48, 64, 4, False, True),
    (4, 3, 2, 64, 128, 6, True, False),
    (6, 3, 1, 128, 160, 9, True, False),
    (6, 3, 2, 160, 256, 15, True, False),
]

def create_model():
    model = EfficientNetV2(layer_info=EFFICIENTNETV2_S_CONFIG, num_classes=1)
    return model

def create_efficientnetv2_backbone(in_channels=3):
    model = EfficientNetV2(layer_info=EFFICIENTNETV2_S_CONFIG, in_channels=in_channels)
    return model

def load_model(path):
    model = create_model()
    model.load_state_dict(torch.load(path))
    return model


if __name__ == '__main__':
    model = create_model()
    test = torch.randn(3, 3, 256, 256)
    result = model(test)
    
    criterion = nn.BCELoss()
    print(result.shape)
    print(criterion(result, torch.ones(3, 1)))

import torch

import torch.nn as nn
import torch.nn.functional as F

import torchvision

class ClassiferModule(nn.Module):
    def __init__(self):
        super(ClassiferModule, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(1280, 1)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x

def create_model():
    model = torchvision.models.efficientnet_v2_s()
    model.classifier = ClassiferModule()
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{num_parameters:,} total parameters.')
    return model

def load_model(path):
    model = create_model()
    model.load_state_dict(torch.load(path))
    return model


if __name__ == '__main__':
    model = create_model()
    test = torch.randn(3, 3, 256, 256)
    result = model(test)
    print(result.shape)
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class weighted_binary_cross_entropy(nn.Module):
    def __init__(self, weight=None):
        super(weighted_binary_cross_entropy, self).__init__()
        self.weight = weight
        if weight is not None:
            self.loss = lambda output, target: torch.neg(torch.mean((weight * (target * torch.log(output))) + (1 * ((1 - target) * torch.log(1 - output)))))
        else:
            self.loss = nn.BCELoss()
    
    def forward(self, output, target):
        loss = self.loss(output, target)
        return loss
    
    def __repr__(self):
        if self.weight is None:
            return "Binary Cross Entropy"
        else:
            return "{self.weight} Weighted Binary Cross Entropy"

class SCLoss(nn.Module):
    def __init__(self, device, dim, weight=None):
        super().__init__()
        self.weight = weight
        if weight is not None:
            self.softmax = lambda output, target: torch.neg(torch.mean((weight * (target * torch.log(output))) + ((1 - target) * torch.log(1 - output))))
        else:
            self.softmax = nn.BCEWithLogitsLoss()
        self.real_center = nn.Parameter(torch.randn(1, dim), requires_grad=True).to(device)
        self.dim = dim
        self.lamb = 0.5
    
    def forward(self, output, target, vectors):
        loss = self.softmax(output, target)
        
        real_indexes = torch.where(target == 0)[0]
        fake_indexes = torch.where(target == 1)[0]
        
        
        real_vectors = vectors[real_indexes]
        fake_vectors = vectors[fake_indexes]
        
        # Calculate Center Loss
        m_real = 1 - torch.mean(F.cosine_similarity(real_vectors, self.real_center, dim=1))
        m_fake = torch.mean(F.cosine_similarity(fake_vectors, self.real_center, dim=1))
        
        if real_indexes.shape[0] == 0:
            m_real = 0  
        if fake_indexes.shape[0] == 0:
            m_fake = 0

        center_loss = m_real + m_fake
        
        return loss + self.lamb * center_loss

    def __repr__(self):
        if self.weight is None:
            return "SCLoss with BCE"
        else:
            return f"{self.weight} Weighted BCE with SCLoss"
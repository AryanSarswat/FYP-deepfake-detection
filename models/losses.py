import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision

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
        
class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = torch.tensor([1 - alpha, alpha]).cuda()
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

    def __repr__(self):
        return f"Focal Loss with alpha={self.alpha} and gamma={self.gamma}"
    
class SCLoss(nn.Module):
    def __init__(self, device, dim, weight=None):
        super().__init__()
        self.weight = weight
        if weight is not None:
            self.softmax = lambda output, target: torch.neg(torch.mean((weight * (target * torch.log(F.sigmoid(output)))) + ((1 - target) * torch.log(1 - F.sigmoid(output)))))
        else:
            self.softmax = nn.BCEWithLogitsLoss()
            
        self.real_center = nn.Parameter(torch.randn(1, dim), requires_grad=True).to(device)
        self.dim = dim
        self.lamb = 0.9
        self.spatial = 0.5
    
    def forward(self, output_t, output_s, target_t, target_s, vectors):
        loss_t = self.softmax(output_t, target_t)
        loss_s = 0.5 * self.softmax(output_s, target_s)
        loss = loss_t + loss_s
        
        real_indexes = torch.where(target_t == 0)[0]
        fake_indexes = torch.where(target_t == 1)[0]

        real_vectors = vectors[real_indexes]
        fake_vectors = vectors[fake_indexes]
        
        # Calculate Center Loss
        m_real = 1 - torch.mean(F.cosine_similarity(real_vectors, self.real_center, dim=1))
        m_fake = torch.mean(F.cosine_similarity(fake_vectors, self.real_center, dim=1))
        
        if len(real_indexes) == 0:
            m_real = 0  
        if len(fake_indexes) == 0:
            m_fake = 0

        center_loss = m_real + m_fake
        
        return loss, self.lamb * center_loss

    def __repr__(self):
        if self.weight is None:
            return "SCLoss with BCE"
        else:
            return f"{self.weight} Weighted BCE with SCLoss"
        

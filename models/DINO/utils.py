import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


class DataAugmentations:
    def __init__(self, global_crops_scale=(0.4, 1), local_crops_scale=(0.05, 0.4), n_local_crop=8, size=224):
        """
        Creates crops of an input image with various augmentations.

        Args:
            global_crops_scale (tuple, optional): Range of sizes for global crops. Defaults to (0.4,1).
            local_crops_scale (tuple, optional): Range of sizes for local crops. Defaults to (0.05, 0.4).
            n_local_crop (int, optional): Number of crops to create. Defaults to 8.
            size (int, optional): size of the final image. Defaults to 224.
        """
        self.n_local_crop = n_local_crop
        
        random_gaussian_blur = lambda p: transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2))],
            p=p,
        )
        
        flip_and_jitter =  transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.8, 
                    contrast=0.8, 
                    saturation=0.8, 
                    hue=0.2)
                ]),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        normalize = transforms.Compose([
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ])
                                       
        self.global_1 = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_jitter,
            random_gaussian_blur(p=1),
            normalize,
        ])
        
        self.global_2 = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_jitter,
            random_gaussian_blur(p=0.1),
            transforms.RandomSolarize(170, p=0.2),
            normalize,
        ])

        self.local = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_jitter,
            random_gaussian_blur(p=0.5),
            normalize,
        ]) 
        
    def __call__(self, x):
        all_crops = []
        all_crops.append(self.global_1(x))
        all_crops.append(self.global_2(x))
        all_crops.extend([self.local(x) for _ in range(self.n_local_crop)])
        
        return all_crops

class Head(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, bottleneck_dim=256, n_layers=3, norm_last_layer=False):
        """
        MLP head for DINO.

        Args:
            in_dim (int): Dimensionality of token embeddings.
            out_dim (int): Dimensionality of output.
            hidden_dim (int, optional): Dimensionality of hidden layer. Defaults to 512.
            bottleneck_dim (int, optional): Dimensionality of second last layer. Defaults to 256.
            n_layers (int, optional): Number of layers. Defaults to 3.
            norm_last_layer (bool, optional): If true last layer's weights are frozen. Defaults to False.
        """
        super().__init__()
        if n_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        
        self.apply(self._init_weights)
        
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
        
    def _init_weights(self, m):
        """Initialize learnable parameters."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.mlp(x)  # (n_samples, bottleneck_dim)
        x = nn.functional.normalize(x, dim=-1, p=2)  # (n_samples, bottleneck_dim)
        x = self.last_layer(x)  # (n_samples, out_dim)

        return x

class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, new_head):
        """
        Wrapper class for DINO.

        Args:
            backbone (_type_): Model to train
            new_head (_type_): Head that will be placed on top of the backbone
        """
        super().__init__()
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.new_head = new_head
    
    def forward(self, x):
        n_crops = len(x)
        concat = torch.cat(x, dim=0) # (n_crops * n_samples, 3, 256, 256)
        cls_embedding = self.backbone(concat) # (n_crops * n_samples, token_dim)
        logits = self.new_head(cls_embedding) # (n_crops * n_samples, out_dim)
        chunks = logits.chunk(n_crops, dim=0) # n_crops * (n_samples, out_dim)
        
        return chunks

class Loss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("centers", torch.zeros(out_dim))
        
    def forward(self, student, teacher):
        n_crops = len(student)
        student = torch.cat(student, dim=0)


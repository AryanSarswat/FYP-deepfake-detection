import math

import cv2
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F
import torchvision
from albumentations import *


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        """
        Computes standard normal cumulative distribution function
        """
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        raise ValueError('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
                         'The distribution of values may be incorrect.')
        
    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
    
        tensor.erfinv_()
        
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def fix_random_seed(seed=31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
def clip_gradients(model, clip_value):
    norms = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                norm = param.grad.data.norm(2)
                norms.append(norm.item())
                clip_coef = clip_value / (norm + 1e-6)
                if clip_coef < 1:
                    param.grad.data.mul_(clip_coef)
    return norms

def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tesnor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    Taken from timm library
    Args:
        x (_type_): _description_
        drop_prob (float, optional): _description_. Defaults to 0..
        training (bool, optional): _description_. Defaults to False.
        scale_by_keep (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """
    Taken from timm library

    Args:
        nn (_type_): _description_
    """
    def __init__(self, drop_prob:None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    

def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
    """
    assert len(clip.size()) == 4, "clip should be a 4D tensor"
    return clip[..., i: i + h, j: j + w]


def resize(clip, target_size, interpolation_mode):
    assert len(target_size) == 2, "target size should be tuple (height, width)"
    return torch.nn.functional.interpolate(
        clip, size=target_size, mode=interpolation_mode
    )

def resized_crop(clip, i, j, h, w, size, interpolation_mode="bilinear"):
    """
    Do spatial cropping and resizing to the video clip
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped region.
        w (int): Width of the cropped region.
        size (tuple(int, int)): height and width of resized clip
    Returns:
        clip (torch.tensor): Resized and cropped clip. Size is (C, T, H, W)
    """
    assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
    clip = crop(clip, i, j, h, w)
    clip = resize(clip, size, interpolation_mode)
    return clip

# Transformations
class DataAugmentationImage:
    def __init__(self, size: int = 224):
        
        self.gaussian_noise = GaussNoise(p=1)
        self.image_compression = ImageCompression(quality_lower=60, quality_upper=100, p=1)
        self.GaussianBlur = GaussianBlur(blur_limit=3, p=1)
        self.HorizontalFlip = HorizontalFlip()
        self.one_of = [RandomBrightnessContrast(p=1), FancyPCA(p=1), HueSaturationValue(p=1)]
        self.grey_scale = ToGray(p=1)
        self.shift_scale_rotate = ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=1)

        self.normalize = torchvision.transforms.ToTensor()
        
        
    def get_transforms(self):
        noise = 1 if np.random.random() < 0.1 else 0
        compression = 1 if np.random.random() < 0.5 else 0
        blur = 1 if np.random.random() < 0.05 else 0
        flip = 1 if np.random.random() < 0.5 else 0
        one_of = [np.random.random() for _ in range(3)]
        one_of = np.argmax(one_of)
        grey = 1 if np.random.random() < 0.2 else 0
        shift = 1 if np.random.random() < 0.5 else 0
        return noise, compression, blur, flip, one_of, grey, shift
    
    def __call__(self, img, noise, compression, blur, flip, one_of, grey, shift):
        #img = PIL.Image.fromarray(img.astype(np.uint8))
        img = self.gaussian_noise(image=img)['image'] if noise == 1 else img
        img = self.image_compression(image=img)['image'] if compression == 1 else img
        img = self.GaussianBlur(image=img)['image'] if blur == 1 else img
        img = self.HorizontalFlip(image=img)['image'] if flip == 1 else img
        inter = self.one_of[one_of]
        img = inter(image=img)['image']
        img = self.grey_scale(image=img)['image'] if grey == 1 else img
        img = self.shift_scale_rotate(image=img)['image'] if shift == 1 else img
        img = self.normalize(img)
        
        return img


class DINOHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, proj_hidden_dim: int = 2048, bottleneck_dim: int = 512, n_layers:int = 3,
                 norm_last_layer: bool = True):
        super(DINOHead, self).__init__()
        if n_layers == 1:
            self.mlp = nn.Linear(in_channels, bottleneck_dim)
        else:
            layers = [nn.Linear(in_channels, proj_hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(proj_hidden_dim, proj_hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(proj_hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
        
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_channels, bias=False))
        
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
    
class MultiCropWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, vary_fr: bool = False) -> None:
        super(MultiCropWrapper, self).__init__()
        
        self.backbone = backbone
        self.head = head
        
    def forward(self, x):
        n_crops = len(x)
        
        concatenated = torch.cat(x, dim=0) # (n_samples * n_crops, C, T, H, W)
        cls_emb = self.backbone(concatenated) # (n_samples * n_crops, C)
        logits = self.head(cls_emb) # (n_samples * n_crops, n_classes)
        chunks = logits.chunk(n_crops) # n_crops * (n_samples, n_classes)
        
        return chunks
    

class DINOLoss(nn.Module):
    def __init__(self, out_dim: int, teacher_temp: float = 0.04, student_temp: float = 0.1, center_momentum: float = 0.9):
        super(DINOLoss, self).__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        
    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor):
        student_temp = [s / self.student_temp for s in student_output]
        teacher_temp = [(t - self.center) / self.teacher_temp for t in teacher_output]
        
        student_sm = [F.log_softmax(s, dim=-1) for s in student_temp]
        teacher_sm = [F.softmax(t, dim=-1).detach() for t in teacher_temp]
        
        total_loss = 0
        n_loss_terms = 0
        
        for t_ix, t in enumerate(teacher_sm):
            for s_ix, s in enumerate(student_sm):
                if t_ix == s_ix:
                    continue
                
                loss = torch.sum(-t * s, dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        total_loss = total_loss / n_loss_terms
        self.update_center(teacher_output)
        
        return total_loss
    
    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True) # (1, out_dim)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
    
if __name__ == "__main__":
    cropper = DataAugmentation(size=224)
    test = np.random.rand(16, 3, 224, 224)
    result = cropper(test)
    print(result.shape)
    

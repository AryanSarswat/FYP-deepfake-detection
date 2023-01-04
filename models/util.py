import math
import torch

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
import torch
from torch import nn


class GlobalResponseNorm3d(nn.Module):
    """Global Response Normalization layer"""

    def __init__(self, dim, eps=1e-6, channels_last=True):
        super().__init__()
        self.eps = eps
        self.channels_last = channels_last

        self.weight = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        if self.channels_last:
            spatial_dim = (1, 2, 3)
            channel_dim = -1
            wb_shape = (1, 1, 1, 1, -1)
        else:
            spatial_dim = (2, 3, 4)
            channel_dim = 1
            wb_shape = (1, -1, 1, 1, 1)

        x_g = x.norm(p=2, dim=spatial_dim, keepdim=True)
        x_n = x_g / (x_g.mean(dim=channel_dim, keepdim=True) + self.eps)
        return x + torch.addcmul(self.bias.view(wb_shape), self.weight.view(wb_shape), x * x_n)

# Pytorch implementation for `SIERRA: A robust bilateral feature upsampler for dense prediction'
# requirements:
# 1.Pytorch
# 2.mmcv
# Usage is set below for directly running this script.

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from mmcv.ops.carafe import carafe
from mmcv.cnn import normal_init, xavier_init


class SIERRA(nn.Module):
    def __init__(self, scale=2, kernel_size=3):
        super().__init__()
        assert isinstance(scale, int) and scale >= 2, \
            'scale must be integers and greater than 2'
        assert isinstance(kernel_size, int) and kernel_size >= 3 and kernel_size % 2 == 1, \
            'kernel size must be odd integers and greater than 3'
        self.scale = scale
        self.kernel_size = kernel_size
        kernel = self.get_kernel()
        self.register_buffer('kernel', kernel)

    def get_kernel(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        center = torch.stack(torch.meshgrid([h, h])).view(2, 1, self.scale ** 2)
        h = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        neighbor = torch.stack(torch.meshgrid([h, h])).view(2, self.kernel_size ** 2, 1)
        kernel = 1 / (torch.sum((center - neighbor) ** 2, dim=0) + 0.2)
        return kernel

    def forward(self, x):
        B, C, H, W = x.shape
        mean = torch.mean(x, dim=1, keepdim=True)
        grad = F.unfold(mean, kernel_size=self.kernel_size,
                        padding=self.kernel_size // 2).view(B, self.kernel_size ** 2, H, W) - mean
        grad = 1 / (grad ** 2 + 0.2).unsqueeze(2)
        kernels = self.kernel.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        kernels = F.softmax(F.pixel_shuffle(grad * kernels, upscale_factor=self.scale).squeeze(2), dim=1)
        return carafe(x, kernels, self.kernel_size, 1, self.scale)


class SIERRASp(nn.Module):
    def __init__(self, scale=2, kernel_size=3):
        super().__init__()
        assert isinstance(scale, int) and scale >= 2, \
            'scale must be integers and greater than 2'
        assert isinstance(kernel_size, int) and kernel_size >= 3 and kernel_size % 2 == 1, \
            'kernel size must be odd integers and greater than 3'
        self.scale = scale
        self.kernel_size = kernel_size
        h = torch.arange((-scale + 1) / 2, (scale - 1) / 2 + 1) / scale
        center = torch.stack(torch.meshgrid([h, h])).view(1, 2 * scale ** 2, 1, 1)
        self.register_buffer('center', center)
        h = torch.arange(-(kernel_size - 1) / 2, (kernel_size - 1) / 2 + 1)
        neighbor = torch.stack(torch.meshgrid([h, h])).view(1, 2 * kernel_size ** 2, 1, 1)
        self.register_buffer('neighbor', neighbor)
        self.offset = nn.Conv2d(1, 2, kernel_size=2 * scale + 1, padding=scale)
        normal_init(self.offset, std=0.001)

    def get_kernel(self, offset):
        B, C, H, W = offset.shape
        center = F.pixel_shuffle(F.interpolate(
            self.center, size=[H // self.scale, W // self.scale]), upscale_factor=self.scale)
        shift = center + offset
        neighbor = F.interpolate(self.neighbor, size=[H, W]).view(1, 2, self.kernel_size ** 2, H, W)
        kernels = 1 / (torch.sum((shift.unsqueeze(2) - neighbor) ** 2, dim=1) + 0.2)
        return kernels

    def forward(self, x):
        B, C, H, W = x.shape
        mean = torch.mean(x, dim=1, keepdim=True)
        offset = torch.tanh(self.offset(F.interpolate(mean, scale_factor=self.scale))) / (2 * self.scale)
        kernels = self.get_kernel(offset)
        grad = F.unfold(mean, kernel_size=self.kernel_size,
                        padding=self.kernel_size // 2).view(B, self.kernel_size ** 2, H, W) - mean
        grad = 1 / (grad ** 2 + 0.2)
        kernels = F.softmax(F.interpolate(grad, scale_factor=self.scale) * kernels, dim=1)
        return carafe(x, kernels, self.kernel_size, 1, self.scale)


class GGBilinear(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = self.get_kernel()
        self.register_buffer('kernel', kernel)

    def get_kernel(self):
        h = torch.from_numpy(np.array([-1/4, 1/4]))
        h, w = torch.meshgrid([h, h])
        h = h.unsqueeze(0).unsqueeze(0)
        w = w.unsqueeze(0).unsqueeze(0)
        bilinear_kernel = torch.cat(
            [(1/2 - h) * (1/2 - w), (1/2 - h) * (1/2 + w),
             (1/2 + h) * (1/2 - w), (1/2 + h) * (1/2 + w)],
            dim=1).view(1, 4, -1, 1, 1)
        return bilinear_kernel

    def forward(self, x):
        B, C, H, W = x.shape
        mean = F.unfold(torch.mean(x, dim=1, keepdim=True),
                        kernel_size=2, padding=1).view(B, 4, H + 1, W + 1)
        grad = 1 / ((mean.unsqueeze(2) - mean.unsqueeze(1)) ** 2 + 0.2)
        kernels = F.softmax(grad * self.kernel, dim=1).unsqueeze(1)
        x = F.unfold(x, kernel_size=2, padding=1).view(B, C, 4, 1, H + 1, W + 1)
        return F.pad(F.pixel_shuffle(torch.sum(kernels * x, dim=2),
                                     upscale_factor=2).squeeze(2), pad=[-1] * 4)


if __name__ == '__main__':
    x = torch.randn(2, 3, 4, 4).to('cuda')
    up = SIERRA().to('cuda')
    print(up(x).shape)

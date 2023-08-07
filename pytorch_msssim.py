import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        cs = cs.mean()
        ret = ssim_map.mean()
    else:
        cs = cs.mean(1).mean(1).mean(1)
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)

        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    # TODO: remove support for normalize == True (kept for backward support)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1]) * pow2[-1]
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


"""
Created on Thu Dec  3 00:28:15 2020

@author: Yunpeng Li, Tianjin University
"""


# class MS_SSIM_L1_LOSS(nn.Module):
#     # Have to use cuda, otherwise the speed is too slow.
#     def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
#                  data_range = 1.0,
#                  K=(0.01, 0.03),
#                  alpha=0.025,
#                  compensation=200.0,
#                  cuda_dev=0,):
#         super(MS_SSIM_L1_LOSS, self).__init__()
#         self.DR = data_range
#         self.C1 = (K[0] * data_range) ** 2
#         self.C2 = (K[1] * data_range) ** 2
#         self.pad = int(2 * gaussian_sigmas[-1])
#         self.alpha = alpha
#         self.compensation=compensation
#         filter_size = int(4 * gaussian_sigmas[-1] + 1)
#         g_masks = torch.zeros((3*len(gaussian_sigmas), 1, filter_size, filter_size))
#         for idx, sigma in enumerate(gaussian_sigmas):
#             # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
#             g_masks[3*idx+0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
#             g_masks[3*idx+1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
#             g_masks[3*idx+2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
#         self.g_masks = g_masks.cuda(cuda_dev)

#     def _fspecial_gauss_1d(self, size, sigma):
#         """Create 1-D gauss kernel
#         Args:
#             size (int): the size of gauss kernel
#             sigma (float): sigma of normal distribution

#         Returns:
#             torch.Tensor: 1D kernel (size)
#         """
#         coords = torch.arange(size).to(dtype=torch.float)
#         coords -= size // 2
#         g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
#         g /= g.sum()
#         return g.reshape(-1)

#     def _fspecial_gauss_2d(self, size, sigma):
#         """Create 2-D gauss kernel
#         Args:
#             size (int): the size of gauss kernel
#             sigma (float): sigma of normal distribution

#         Returns:
#             torch.Tensor: 2D kernel (size x size)
#         """
#         gaussian_vec = self._fspecial_gauss_1d(size, sigma)
#         return torch.outer(gaussian_vec, gaussian_vec)

#     def forward(self, x, y):
#         b, c, h, w = x.shape
#         mux = F.conv2d(x, self.g_masks, groups=3, padding=self.pad)
#         muy = F.conv2d(y, self.g_masks, groups=3, padding=self.pad)

#         mux2 = mux * mux
#         muy2 = muy * muy
#         muxy = mux * muy

#         sigmax2 = F.conv2d(x * x, self.g_masks, groups=3, padding=self.pad) - mux2
#         sigmay2 = F.conv2d(y * y, self.g_masks, groups=3, padding=self.pad) - muy2
#         sigmaxy = F.conv2d(x * y, self.g_masks, groups=3, padding=self.pad) - muxy

#         # l(j), cs(j) in MS-SSIM
#         l  = (2 * muxy    + self.C1) / (mux2    + muy2    + self.C1)  # [B, 15, H, W]
#         cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

#         lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
#         PIcs = cs.prod(dim=1)

#         loss_ms_ssim = 1 - lM*PIcs  # [B, H, W]

#         loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
#         # average l1 loss in 3 channels
#         gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-3, length=3),
#                                groups=3, padding=self.pad).mean(1)  # [B, H, W]

#         loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
#         loss_mix = self.compensation*loss_mix

#         return loss_mix.mean()

# import torch
# import torch.nn.functional as F
# from torch.autograd import Variable
# import numpy as np
# from math import exp

# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
#     return gauss/gauss.sum()

# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#     return window

# def _ssim(img1, img2, window, window_size, channel, size_average = True):
#     mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
#     mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1*mu2

#     sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
#     sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

#     C1 = 0.01**2
#     C2 = 0.03**2

#     ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)

# class SSIM(torch.nn.Module):
#     def __init__(self, window_size = 11, size_average = True):
#         super(SSIM, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.channel = 1
#         self.window = create_window(window_size, self.channel)

#     def forward(self, img1, img2):
#         (_, channel, _, _) = img1.size()

#         if channel == self.channel and self.window.data.type() == img1.data.type():
#             window = self.window
#         else:
#             window = create_window(self.window_size, channel)
            
#             if img1.is_cuda:
#                 window = window.cuda(img1.get_device())
#             window = window.type_as(img1)
            
#             self.window = window
#             self.channel = channel


#         return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

# def ssim(img1, img2, window_size = 11, size_average = True):
#     (_, channel, _, _) = img1.size()
#     window = create_window(window_size, channel)
    
#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)
    
#     return _ssim(img1, img2, window, window_size, channel, size_average)
import torch
import torch.nn as nn
from torch.nn import ReLU, LeakyReLU
import torch.nn.functional as F
import math
import numpy as np
from .base_networks import (convolve, convolveLeakyReLU, upconvolve, upconvolveLeakyReLU, ConvReg)
from .spatial_transformer import SpatialTransform


def affine_flow(W, b, sd, sh, sw):
    # W: (B, 3, 3), b: (B, 3), len = 128
    device = W.device
    b = b.view([-1, 3, 1, 1, 1])

    xr = torch.arange(-(sw - 1) / 2.0, sw / 2.0, 1.0, dtype=torch.float32)
    xr = xr.view([1, 1, 1, 1, -1]).to(device)
    yr = torch.arange(-(sh - 1) / 2.0, sh / 2.0, 1.0, dtype=torch.float32)
    yr = yr.view([1, 1, 1, -1, 1]).to(device)
    zr = torch.arange(-(sd - 1) / 2.0, sd / 2.0, 1.0, dtype=torch.float32)
    zr = zr.view([1, 1, -1, 1, 1]).to(device)

    wx = W[:, :, 0]
    wx = wx.view([-1, 3, 1, 1, 1])
    wy = W[:, :, 1]
    wy = wy.view([-1, 3, 1, 1, 1])
    wz = W[:, :, 2]
    wz = wz.view([-1, 3, 1, 1, 1])
    return xr * wx + yr * wy + zr * wz + b


class NIL(nn.Module):
    def __init__(self, dim=3, channels=8, flow_multiplier=1.):
        super(NIL, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.dim = dim
        # self.im_size = im_size

        self.conv1 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(2, channels, 3, 2, padding=1),
        )
        self.conv2 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(channels, channels * 2, 3, 2, padding=1),
        )
        self.conv3 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(channels * 2, channels * 4, 3, 2, padding=1),
        )
        self.conv4 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(channels * 4, channels * 8, 3, 2, padding=1),
        )
        self.conv5 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(channels * 8, channels * 16, 3, 2, padding=1),
        )
        self.conv6 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(channels * 16, channels * 32, 3, 2, padding=1),
        )

        ### without mid layer

        self.decode6 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(channels * 32, channels * 16, 3, 1, padding=1),
        )
        self.decode5 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(channels * 32, channels * 8, 3, 1, padding=1),
        )
        self.decode4 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(channels * 16, channels * 4, 3, 1, padding=1),
        )
        self.decode3 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(channels * 8, channels * 2, 3, 1, padding=1),
        )
        self.decode2 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(channels * 4, channels, 3, 1, padding=1),
        )
        self.decode1 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(channels * 2, channels, 3, 1, padding=1),
        )

        self.flow = nn.Sequential(
            nn.Conv3d(channels, 3, 3, 1, padding=1),
            nn.Upsample(scale_factor=2, mode="trilinear")
        )
        self.up_nearest = nn.Upsample(scale_factor=2, mode="nearest")

        self.convreg6 = ConvReg(256, 512 + 3)
        self.convreg5 = ConvReg(128, 256 + 3)
        self.convreg4 = ConvReg(64, 128 + 3)
        self.convreg3 = ConvReg(32, 64 + 3)
        self.convreg2 = ConvReg(16, 32 + 3)

    def forward(self, fixed, moving, **kwargs):
        concat_image = torch.cat((fixed, moving), dim=1)
        c1 = self.conv1(concat_image)  # C8, 64
        c2 = self.conv2(c1)  # C16, 32
        c3 = self.conv3(c2)  # C32, 16
        c4 = self.conv4(c3)  # C64, 8
        c5 = self.conv5(c4)  # C128, 4
        c6 = self.conv6(c5)  # C256, 2

        d6 = self.decode6(c6)  # C128, 2
        concat6 = torch.cat((self.up_nearest(d6), c5), dim=1)
        d5 = self.decode5(concat6)  # C64, 4
        concat5 = torch.cat((self.up_nearest(d5), c4), dim=1)  # C128, 8
        d4 = self.decode4(concat5)  # C32, 8
        concat4 = torch.cat((self.up_nearest(d4), c3), dim=1)  # C64, 16
        d3 = self.decode3(concat4)  # C16, 16
        concat3 = torch.cat((self.up_nearest(d3), c2), dim=1)  # C32, 32
        d2 = self.decode2(concat3)  # C8, 32
        concat2 = torch.cat((self.up_nearest(d2), c1), dim=1)  # C16, 64
        d1 = self.decode1(concat2)  # C8, 64
        net = self.flow(d1)  # C3, 128

        reg6 = self.convreg6(concat6, flow=kwargs.get("flow", None))
        reg5 = self.convreg5(concat5, flow=kwargs.get("flow", None))
        reg4 = self.convreg4(concat4, flow=kwargs.get("flow", None))
        reg3 = self.convreg3(concat3, flow=kwargs.get("flow", None))
        reg2 = self.convreg2(concat2, flow=kwargs.get("flow", None))
        decoder_feature_maps = [reg6, reg5, reg4, reg3, reg2]

        return {
            "flow": net * self.flow_multiplier,
            "decoder_regs": decoder_feature_maps,
        }


class NIL_lowres(nn.Module):
    def __init__(self, dim=3, channels=8, flow_multiplier=1.):
        super(NIL_lowres, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.dim = dim
        # self.im_size = im_size

        self.conv1 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(2, channels, 3, 2, padding=1),
        )
        self.conv2 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(channels, channels * 2, 3, 2, padding=1),
        )
        self.conv3 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(channels * 2, channels * 4, 3, 2, padding=1),
        )
        self.conv4 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(channels * 4, channels * 8, 3, 2, padding=1),
        )
        self.conv5 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(channels * 8, channels * 16, 3, 2, padding=1),
        )

        ### without mid layer

        self.decode5 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(channels * 16, channels * 8, 3, 1, padding=1),
        )
        self.decode4 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(channels * 16, channels * 4, 3, 1, padding=1),
        )
        self.decode3 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(channels * 8, channels * 2, 3, 1, padding=1),
        )
        self.decode2 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(channels * 4, channels, 3, 1, padding=1),
        )
        self.decode1 = nn.Sequential(
            LeakyReLU(0.1),
            nn.Conv3d(channels * 2, channels, 3, 1, padding=1),
        )

        self.flow = nn.Sequential(
            nn.Conv3d(channels, 3, 3, 1, padding=1),
            nn.Upsample(scale_factor=2, mode="trilinear")
        )
        self.up_nearest = nn.Upsample(scale_factor=2, mode="nearest")

        self.convreg6 = ConvReg(256, 512 + 3)
        self.convreg5 = ConvReg(128, 256 + 3)
        self.convreg4 = ConvReg(64, 128 + 3)
        self.convreg3 = ConvReg(32, 64 + 3)
        self.convreg2 = ConvReg(16, 32 + 3)

    def forward(self, fixed, moving, **kwargs):
        concat_image = torch.cat((fixed, moving), dim=1)
        c1 = self.conv1(concat_image)  # C8, 64
        c2 = self.conv2(c1)  # C16, 32
        c3 = self.conv3(c2)  # C32, 16
        c4 = self.conv4(c3)  # C64, 8
        c5 = self.conv5(c4)  # C128, 4

        d5 = self.decode5(c5)  # C64, 4
        concat5 = torch.cat((self.up_nearest(d5), c4), dim=1)  # C128, 8
        d4 = self.decode4(concat5)  # C32, 8
        concat4 = torch.cat((self.up_nearest(d4), c3), dim=1)  # C64, 16
        d3 = self.decode3(concat4)  # C16, 16
        concat3 = torch.cat((self.up_nearest(d3), c2), dim=1)  # C32, 32
        d2 = self.decode2(concat3)  # C8, 32
        concat2 = torch.cat((self.up_nearest(d2), c1), dim=1)  # C16, 64
        d1 = self.decode1(concat2)  # C8, 64
        net = self.flow(d1)  # C3, 128

        # reg5 = self.convreg5(concat5, flow=kwargs.get("flow", None))
        # reg4 = self.convreg4(concat4, flow=kwargs.get("flow", None))
        # reg3 = self.convreg3(concat3, flow=kwargs.get("flow", None))
        # reg2 = self.convreg2(concat2, flow=kwargs.get("flow", None))
        # decoder_feature_maps = [None, reg5, reg4, reg3, reg2]

        return {
            "flow": net * self.flow_multiplier,
            # "decoder_regs": decoder_feature_maps,
        }

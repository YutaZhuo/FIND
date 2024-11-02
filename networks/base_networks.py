import torch
import torch.nn as nn
from torch.nn import ReLU, LeakyReLU
import torch.nn.functional as F
import math
import numpy as np
from .spatial_transformer import SpatialTransform
from .ULAE import Uncoupled_Encoding_Block, ResizeTransform, Accumulative_Enhancement_1, Accumulative_Enhancement_2, \
    Accumulative_Enhancement_3


def convolve(in_channels, out_channels, kernel_size, stride, dim=3):
    # through verification, the padding surely is 1, as input_size is even, kernel=3, stride=1 or 2
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=1)


def convolveReLU(in_channels, out_channels, kernel_size, stride, dim=3):
    # the seq of conv and activation is reverse to origin paper
    return nn.Sequential(ReLU(), convolve(in_channels, out_channels, kernel_size, stride, dim=dim))


def convolveLeakyReLU(in_channels, out_channels, kernel_size, stride, dim=3):
    # the seq of conv and activation is reverse to origin paper
    return nn.Sequential(LeakyReLU(0.1), convolve(in_channels, out_channels, kernel_size, stride, dim=dim))


def upconvolve(in_channels, out_channels, kernel_size, stride, dim=3):
    # through verification, the padding surely is 1
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding=1)


def upconvolveReLU(in_channels, out_channels, kernel_size, stride, dim=3):
    # the seq of conv and activation is reverse to origin paper
    return nn.Sequential(ReLU(), upconvolve(in_channels, out_channels, kernel_size, stride, dim=dim))


def upconvolveLeakyReLU(in_channels, out_channels, kernel_size, stride, dim=3):
    # the seq of conv and activation is reverse to origin paper
    return nn.Sequential(LeakyReLU(0.1), upconvolve(in_channels, out_channels, kernel_size, stride, dim=dim))


def convRelu_dis(in_channels, out_channels, kernel_size, stride, padding):
    conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
    for name, param in conv.named_parameters():
        if "weight" in name:
            nn.init.trunc_normal_(param, std=0.02)
        elif "bias" in name:
            nn.init.zeros_(param)
    return nn.Sequential(conv, LeakyReLU(0.2))


def linear_dis(in_features, out_features):
    linear = nn.Linear(in_features, out_features)
    for name, param in linear.named_parameters():
        if "weight" in name:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.zeros_(param)
    return linear


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


def det3x3(M):
    # M: (B, 3, 3)
    M = [[M[:, i, j] for j in range(3)] for i in range(3)]
    return torch.sum(torch.stack([
        M[0][0] * M[1][1] * M[2][2],
        M[0][1] * M[1][2] * M[2][0],
        M[0][2] * M[1][0] * M[2][1]
    ], dim=0), dim=0) - torch.sum(torch.stack([
        M[0][0] * M[1][2] * M[2][1],
        M[0][1] * M[1][0] * M[2][2],
        M[0][2] * M[1][1] * M[2][0]
    ], dim=0), dim=0)


class feature_fusion(nn.Module):
    """
    Feature Fusion in CRD
    """

    def __init__(self, ch_a, ch_b):
        super(feature_fusion, self).__init__()
        self.conv1 = nn.Conv3d(ch_a + ch_b, ch_a + ch_b, 3, 1, padding=1)
        self.conv2 = nn.Conv3d(ch_a + ch_b, ch_a + ch_b, 3, 1, padding=1)
        self.conv3 = nn.Conv3d(ch_a + ch_b, ch_a + ch_b, 3, 1, padding=1)
        self.conv4 = nn.Conv3d(ch_a + ch_b, ch_b, 3, 1, padding=1)

    def forward(self, input_a, input_b):
        x = torch.cat([input_a, input_b], dim=1)
        x = nn.ReLU(self.conv1(x))
        x = nn.ReLU(self.conv2(x))
        x = nn.ReLU(self.conv3(x))
        x = nn.ReLU(self.conv4(x))
        return x


class VTN(nn.Module):
    def __init__(self, dim=3, flow_multiplier=1., channels=16):
        super(VTN, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.dim = dim

        # Network architecture
        # The first convolution's input is the concatenated image
        self.conv1 = convolveLeakyReLU(2, channels, 3, 2, dim=dim)
        self.conv2 = convolveLeakyReLU(channels, 2 * channels, 3, 2, dim=dim)
        self.conv3 = convolveLeakyReLU(2 * channels, 4 * channels, 3, 2, dim=dim)
        self.conv3_1 = convolveLeakyReLU(4 * channels, 4 * channels, 3, 1, dim=dim)
        self.conv4 = convolveLeakyReLU(4 * channels, 8 * channels, 3, 2, dim=dim)
        self.conv4_1 = convolveLeakyReLU(8 * channels, 8 * channels, 3, 1, dim=dim)
        self.conv5 = convolveLeakyReLU(8 * channels, 16 * channels, 3, 2, dim=dim)
        self.conv5_1 = convolveLeakyReLU(16 * channels, 16 * channels, 3, 1, dim=dim)
        self.conv6 = convolveLeakyReLU(16 * channels, 32 * channels, 3, 2, dim=dim)
        self.conv6_1 = convolveLeakyReLU(32 * channels, 32 * channels, 3, 1, dim=dim)

        self.pred6 = convolve(32 * channels, dim, 3, 1, dim=dim)
        self.upsamp6to5 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv5 = upconvolveLeakyReLU(32 * channels, 16 * channels, 4, 2, dim=dim)

        self.pred5 = convolve(32 * channels + dim, dim, 3, 1, dim=dim)  # 514 = 32 * channels + 1 + 1
        self.upsamp5to4 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv4 = upconvolveLeakyReLU(32 * channels + dim, 8 * channels, 4, 2, dim=dim)

        self.pred4 = convolve(16 * channels + dim, dim, 3, 1, dim=dim)  # 258 = 64 * channels + 1 + 1
        self.upsamp4to3 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv3 = upconvolveLeakyReLU(16 * channels + dim, 4 * channels, 4, 2, dim=dim)

        self.pred3 = convolve(8 * channels + dim, dim, 3, 1, dim=dim)
        self.upsamp3to2 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv2 = upconvolveLeakyReLU(8 * channels + dim, 2 * channels, 4, 2, dim=dim)

        self.pred2 = convolve(4 * channels + dim, dim, 3, 1, dim=dim)
        self.upsamp2to1 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv1 = upconvolveLeakyReLU(4 * channels + dim, channels, 4, 2, dim=dim)

        self.pred0 = upconvolve(2 * channels + dim, dim, 4, 2, dim=dim)

    def forward(self, fixed, moving, **kwargs):
        concat_image = torch.cat((fixed, moving), dim=1)
        x1 = self.conv1(concat_image)  # 64
        x2 = self.conv2(x1)  # 32
        x3 = self.conv3(x2)  # 16
        x3_1 = self.conv3_1(x3)
        x4 = self.conv4(x3_1)  # C128, 8
        x4_1 = self.conv4_1(x4)
        x5 = self.conv5(x4_1)  # C256, 4
        x5_1 = self.conv5_1(x5)
        x6 = self.conv6(x5_1)  # 2
        x6_1 = self.conv6_1(x6)  # C512, 2

        pred6 = self.pred6(x6_1)  # 2
        upsamp6to5 = self.upsamp6to5(pred6)
        deconv5 = self.deconv5(x6_1)
        concat5 = torch.cat([x5_1, deconv5, upsamp6to5], dim=1)  # C(512+3), 4

        pred5 = self.pred5(concat5)
        upsamp5to4 = self.upsamp5to4(pred5)
        deconv4 = self.deconv4(concat5)
        concat4 = torch.cat([x4_1, deconv4, upsamp5to4], dim=1)  # C(256+3), 8

        pred4 = self.pred4(concat4)
        upsamp4to3 = self.upsamp4to3(pred4)
        deconv3 = self.deconv3(concat4)
        concat3 = torch.cat([x3_1, deconv3, upsamp4to3], dim=1)  # C(128+3), 16

        pred3 = self.pred3(concat3)
        upsamp3to2 = self.upsamp3to2(pred3)
        deconv2 = self.deconv2(concat3)
        concat2 = torch.cat([x2, deconv2, upsamp3to2], dim=1)  # C(64+3), 32

        pred2 = self.pred2(concat2)
        upsamp2to1 = self.upsamp2to1(pred2)
        deconv1 = self.deconv1(concat2)
        concat1 = torch.cat([x1, deconv1, upsamp2to1], dim=1)  # C(32+3), 64

        pred0 = self.pred0(concat1)  # 128

        decoder_feature_maps = [concat5, concat4, concat3, concat2, concat1]

        return {
            "flow": pred0 * 20 * self.flow_multiplier,
            "decoder_feas": decoder_feature_maps,
        }


class VTN_lowres(nn.Module):
    def __init__(self, dim=3, flow_multiplier=1., channels=16):
        super(VTN_lowres, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.dim = dim

        # Network architecture
        # The first convolution's input is the concatenated image
        self.conv1 = convolveLeakyReLU(2, channels, 3, 2, dim=dim)
        self.conv2 = convolveLeakyReLU(channels, 2 * channels, 3, 2, dim=dim)
        self.conv3 = convolveLeakyReLU(2 * channels, 4 * channels, 3, 2, dim=dim)
        self.conv3_1 = convolveLeakyReLU(4 * channels, 4 * channels, 3, 1, dim=dim)
        self.conv4 = convolveLeakyReLU(4 * channels, 8 * channels, 3, 2, dim=dim)
        self.conv4_1 = convolveLeakyReLU(8 * channels, 8 * channels, 3, 1, dim=dim)
        self.conv5 = convolveLeakyReLU(8 * channels, 16 * channels, 3, 2, dim=dim)
        self.conv5_1 = convolveLeakyReLU(16 * channels, 16 * channels, 3, 1, dim=dim)

        self.pred5 = convolve(16 * channels, dim, 3, 1, dim=dim)  # 514 = 32 * channels + 1 + 1
        self.upsamp5to4 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv4 = upconvolveLeakyReLU(16 * channels, 8 * channels, 4, 2, dim=dim)

        self.pred4 = convolve(16 * channels + dim, dim, 3, 1, dim=dim)  # 258 = 64 * channels + 1 + 1
        self.upsamp4to3 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv3 = upconvolveLeakyReLU(16 * channels + dim, 4 * channels, 4, 2, dim=dim)

        self.pred3 = convolve(8 * channels + dim, dim, 3, 1, dim=dim)
        self.upsamp3to2 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv2 = upconvolveLeakyReLU(8 * channels + dim, 2 * channels, 4, 2, dim=dim)

        self.pred2 = convolve(4 * channels + dim, dim, 3, 1, dim=dim)
        self.upsamp2to1 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv1 = upconvolveLeakyReLU(4 * channels + dim, channels, 4, 2, dim=dim)

        self.pred0 = upconvolve(2 * channels + dim, dim, 4, 2, dim=dim)

    def forward(self, fixed, moving, **kwargs):
        concat_image = torch.cat((fixed, moving), dim=1)
        x1 = self.conv1(concat_image)  # 64
        x2 = self.conv2(x1)  # 32
        x3 = self.conv3(x2)  # 16
        x3_1 = self.conv3_1(x3)
        x4 = self.conv4(x3_1)  # C128, 8
        x4_1 = self.conv4_1(x4)
        x5 = self.conv5(x4_1)  # C256, 4
        x5_1 = self.conv5_1(x5)

        pred5 = self.pred5(x5_1)
        upsamp5to4 = self.upsamp5to4(pred5)
        deconv4 = self.deconv4(x5_1)
        concat4 = torch.cat([x4_1, deconv4, upsamp5to4], dim=1)  # C(256+3), 8

        pred4 = self.pred4(concat4)
        upsamp4to3 = self.upsamp4to3(pred4)
        deconv3 = self.deconv3(concat4)
        concat3 = torch.cat([x3_1, deconv3, upsamp4to3], dim=1)  # C(128+3), 16

        pred3 = self.pred3(concat3)
        upsamp3to2 = self.upsamp3to2(pred3)
        deconv2 = self.deconv2(concat3)
        concat2 = torch.cat([x2, deconv2, upsamp3to2], dim=1)  # C(64+3), 32

        pred2 = self.pred2(concat2)
        upsamp2to1 = self.upsamp2to1(pred2)
        deconv1 = self.deconv1(concat2)
        concat1 = torch.cat([x1, deconv1, upsamp2to1], dim=1)  # C(32+3), 64

        pred0 = self.pred0(concat1)  # 128

        decoder_feature_maps = [None, concat4, concat3, concat2, concat1]

        return {
            "flow": pred0 * 20 * self.flow_multiplier,
            "decoder_feas": decoder_feature_maps,
        }


class ConvReg(nn.Module):
    """Convolutional regression"""

    def __init__(self, s_C, t_C):
        super(ConvReg, self).__init__()
        # MGD
        self.conv1 = nn.Conv3d(s_C, t_C, 1)
        self.convK3_1 = nn.Conv3d(t_C, t_C, 3, padding=1)
        self.convK3_2 = nn.Conv3d(t_C, t_C, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, **kwargs):
        # x = self.conv2(self.relu(self.conv1(x)))
        # x = self.conv2(self.conv1(x))
        # return x

        # MGD
        x1 = self.conv1(x)
        B, C, H, W, D = x1.shape
        p_value = 16
        flow = kwargs.get("flow", None)
        if flow is not None:
            flow = nn.Upsample(scale_factor=H / 128, mode="trilinear")(flow)
            tmp = torch.sqrt(torch.sum(torch.square(flow), dim=1))
            # tmp_max = torch.max(tmp.view(B, -1), dim=1)[0].reshape(B, -1).unsqueeze(2).unsqueeze(3)
            # tmp_min = torch.min(tmp.view(B, -1), dim=1)[0].reshape(B, -1).unsqueeze(2).unsqueeze(3)
            # prob = (1 - ((tmp - tmp_min) / (tmp_max - tmp_min)).unsqueeze(1)).repeat(1, C, 1, 1, 1) * 0.6
            scaler = torch.topk(tmp.view(B, -1), int(tmp.numel() / B * 0.4), dim=1, sorted=True)[0][:, -1] \
                .view(B, 1, 1, 1)
            # scaler = 16
            tmp = (tmp / scaler).unsqueeze(1)
            prob = (1 - F.normalize(tmp, p=p_value, dim=[2, 3, 4])).repeat(1, C, 1, 1, 1)
            assert not (prob.isnan()).all()
            bernoulli = torch.bernoulli(prob).to(x.device)
            assert not torch.sum(bernoulli) == tmp.numel() * C
            # print(torch.sum(bernoulli))
        else:
            prob = torch.full(size=(B, C, H, W, D), fill_value=0.6, device=x.device)
            bernoulli = torch.bernoulli(prob)
        x2 = x1 * bernoulli
        assert x2.shape == x1.shape
        x3 = self.convK3_2(self.relu(self.convK3_1(x2)))
        return x3

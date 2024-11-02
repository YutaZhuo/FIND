import torch
from torch.nn import MSELoss
import torch.nn.functional as F
from networks.base_networks import *


def pearson_correlation(fixed, warped):
    flatten_fixed = torch.flatten(fixed, start_dim=1)
    flatten_warped = torch.flatten(warped, start_dim=1)

    mean1 = torch.reshape(torch.mean(flatten_fixed, dim=-1), [-1, 1])
    mean2 = torch.reshape(torch.mean(flatten_warped, dim=-1), [-1, 1])
    var1 = torch.mean(torch.square(flatten_fixed - mean1), dim=-1)
    var2 = torch.mean(torch.square(flatten_warped - mean2), dim=-1)
    cov12 = torch.mean((flatten_fixed - mean1) * (flatten_warped - mean2), dim=-1)
    pearson_r = cov12 / torch.sqrt((var1 + 1e-6) * (var2 + 1e-6))

    raw_loss = 1 - pearson_r
    raw_loss = torch.mean(raw_loss)

    return raw_loss


def regularize_loss(flow):
    """
    flow has shape (batch, 3, , , )
    """
    lossF = MSELoss(reduction="none")
    flow = flow
    dy = torch.sum(lossF(flow[:, :, 1:, :, :], flow[:, :, :-1, :, :]), dim=[1, 2, 3, 4]) / 2
    dx = torch.sum(lossF(flow[:, :, :, 1:, :], flow[:, :, :, :-1, :]), dim=[1, 2, 3, 4]) / 2
    dz = torch.sum(lossF(flow[:, :, :, :, 1:], flow[:, :, :, :, :-1]), dim=[1, 2, 3, 4]) / 2

    d = torch.mean(dx + dy + dz)

    return d / torch.prod(torch.Tensor(list(flow.shape[1:5])))


def dice_loss(fixed_mask, warped):
    """
    Dice similirity loss
    """

    epsilon = 1e-6

    flat_mask = torch.flatten(fixed_mask, start_dim=1)
    flat_warp = torch.abs(torch.flatten(warped, start_dim=1))
    intersection = torch.sum(flat_mask * flat_warp)
    denominator = torch.sum(flat_mask) + torch.sum(flat_warp) + epsilon
    dice = (2.0 * intersection + epsilon) / denominator

    return 1 - dice


def elem_sym_polys_of_eigen_values(M):
    # M: (B, 3, 3)
    M = [[M[:, i, j] for j in range(3)] for i in range(3)]
    sigma1 = torch.sum(torch.stack([M[0][0], M[1][1], M[2][2]], dim=0), dim=0)
    sigma2 = torch.sum(torch.stack([
        M[0][0] * M[1][1],
        M[1][1] * M[2][2],
        M[2][2] * M[0][0]
    ], dim=0), dim=0) - torch.sum(torch.stack([
        M[0][1] * M[1][0],
        M[1][2] * M[2][1],
        M[2][0] * M[0][2]
    ], dim=0), dim=0)
    sigma3 = torch.sum(torch.stack([
        M[0][0] * M[1][1] * M[2][2],
        M[0][1] * M[1][2] * M[2][0],
        M[0][2] * M[1][0] * M[2][1]
    ], dim=0), dim=0) - torch.sum(torch.stack([
        M[0][0] * M[1][2] * M[2][1],
        M[0][1] * M[1][0] * M[2][2],
        M[0][2] * M[1][1] * M[2][0]
    ], dim=0), dim=0)
    return sigma1, sigma2, sigma3


def det_ortho_loss(W):
    I = torch.eye(3).reshape(1, 3, 3).cuda()
    A = W + I
    det = det3x3(A)  # det: (B)
    lossF = MSELoss(reduction="sum")
    det_loss = lossF(det, torch.ones(det.shape[0]).cuda()) / 2
    eps = 1e-5
    epsI = torch.Tensor([[[eps * elem for elem in row] for row in Mat] for Mat in I]).cuda()
    C = torch.matmul(A.mT, A) + epsI

    s1, s2, s3 = elem_sym_polys_of_eigen_values(C)
    ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)  # (B)
    ortho_loss = torch.sum(ortho_loss)
    return det_loss, ortho_loss



def total_loss(fixed, moving, flows):
    sim_loss = pearson_correlation(fixed, moving)
    # Regularize all flows
    reg_loss = sum([regularize_loss(flow) for flow in flows])
    return sim_loss + reg_loss


def total_loss_forRCN(stem_results, hyper, reference_img, reconstruction, moving=None, deep_sup=False, sep_reg=False):
    """
    Only supervise the final warped img. And regularize all middle stage flows.
    Turn on deep_sup, will also return the raw_loss of middle stage flows.
    """
    det_f = hyper["det"]
    ortho_f = hyper["ortho"]
    reg_f = hyper["reg"]
    img_size = reference_img.shape[2:]
    deep_sup_loss = []
    deep_sup_affine = None

    stem_len = len(stem_results)
    for i, block_result in enumerate(stem_results):
        if "W" in block_result:
            # Affine block
            block_result["det_loss"], block_result["ortho_loss"] = det_ortho_loss(block_result["W"])
            block_result["loss"] = block_result["det_loss"] * det_f \
                                   + block_result["ortho_loss"] * ortho_f
            if i == stem_len - 1:
                # if the Affine block is the final block
                warped = block_result["warped"]
                block_result["raw_loss"] = pearson_correlation(reference_img, warped)
                block_result["loss"] = block_result["loss"] + block_result["raw_loss"]
            elif deep_sup:
                deep_sup_affine = pearson_correlation(reference_img, block_result["warped"])
                # deep_sup_affine = pearson_correlation(reference_img, block_result["extra_warped"])
        else:
            # Deform block
            if i == stem_len - 1:
                # if the current Deformable block is the final block
                warped = block_result["warped"]
                block_result["raw_loss"] = pearson_correlation(reference_img, warped)
            elif deep_sup:
                # deep_sup_loss.append(pearson_correlation(reference_img, block_result["extra_warped"]))
                deep_sup_loss.append(pearson_correlation(reference_img, block_result["warped"]))
            block_result["reg_loss"] = regularize_loss(block_result["flow"]) * reg_f


    if not sep_reg:
        for i, block_result in enumerate(stem_results):
            assert "W" not in block_result
            block_result["loss"] = sum([block_result[k] for k in block_result if k.endswith("loss")])
        loss = sum([r["loss"] for r in stem_results])  # loss is the target to optimize
        if deep_sup:
            return loss, deep_sup_loss, deep_sup_affine
        else:
            return loss
    else:
        loss_raw = sum([r["raw_loss"] for r in stem_results if "raw_loss" in r])  # loss is the target to optimize
        loss_reg = sum([r["reg_loss"] for r in stem_results if "reg_loss" in r])
        if deep_sup:
            return loss_raw, loss_reg, deep_sup_loss, deep_sup_affine
        else:
            return loss_raw, loss_reg


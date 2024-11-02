import torch.nn
from torch.quantization import prepare_qat
from .base_networks import *
from .LightWeight import *
from .spatial_transformer import warp3D, SpatialTransform
from metrics.losses import pearson_correlation, regularize_loss


def mask_metrics(seg1, seg2):
    ''' Given two segmentation seg1, seg2, 0 for background 255 for foreground.
    Calculate the Dice score
    $ 2 * | seg1 \cap seg2 | / (|seg1| + |seg2|) $
    and the Jacc score
    $ | seg1 \cap seg2 | / (|seg1 \cup seg2|) $
    '''
    sizes = torch.prod(torch.tensor(seg1.shape[1:]))
    seg1 = seg1.reshape(-1, sizes)
    seg2 = seg2.reshape(-1, sizes)
    dice_score = 2.0 * torch.sum(seg1 * seg2, dim=-1) / (
            torch.sum(seg1, dim=-1) + torch.sum(seg2, dim=-1))
    union = torch.sum(torch.max(seg1, seg2), dim=-1)
    return (dice_score, torch.sum(
        torch.min(seg1, seg2), dim=-1) / torch.max(union, torch.tensor(0.01)))


def mask_metrics_multilabels(img_gt, img_pred):
    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))
    Dice = []
    Jacc = []
    for c in range(1, 36):
        gt_c_i = img_gt.clone()
        gt_c_i[gt_c_i != c] = 0

        pred_c_i = img_pred.clone()
        pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i = torch.clip(gt_c_i, 0, 1)
        pred_c_i = torch.clip(pred_c_i, 0, 1)
        # Compute the Dice
        top = 2 * torch.sum(torch.logical_and(pred_c_i, gt_c_i))
        bottom = torch.sum(pred_c_i) + torch.sum(gt_c_i)
        bottom = torch.maximum(bottom, torch.tensor(0.01))  # add epsilon.
        dice = top / bottom
        # Compute the Jacc
        top = torch.sum(torch.logical_and(pred_c_i, gt_c_i))
        bottom = torch.sum(torch.logical_or(pred_c_i, gt_c_i))
        bottom = torch.maximum(bottom, torch.tensor(0.01))  # add epsilon.
        jacc = top / bottom

        Dice = Dice + [dice]
        Jacc = Jacc + [jacc]

    # print("Dice:", Dice)
    # print("Jacc:", Jacc)
    return torch.mean(torch.tensor(Dice)), torch.mean(torch.tensor(Jacc))


def mask_metrics_ACDC(img_gt, img_pred):
    # 3-LV, 2-Myo, 1-RV
    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))
    Dice = []
    Jacc = []
    for c in [3, 2, 4, 1, 5]:
        gt_c_i = img_gt.clone()
        if c == 4:
            gt_c_i[gt_c_i == 2] = c
            gt_c_i[gt_c_i == 3] = c
        elif c == 5:
            gt_c_i[gt_c_i > 0] = c
        gt_c_i[gt_c_i != c] = 0

        pred_c_i = img_pred.clone()
        if c == 4:
            pred_c_i[pred_c_i == 2] = c
            pred_c_i[pred_c_i == 3] = c
        elif c == 5:
            pred_c_i[pred_c_i > 0] = c
        pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i = torch.clip(gt_c_i, 0, 1)
        pred_c_i = torch.clip(pred_c_i, 0, 1)
        # Compute the Dice
        top = 2 * torch.sum(torch.logical_and(pred_c_i, gt_c_i))
        bottom = torch.sum(pred_c_i) + torch.sum(gt_c_i)
        bottom = torch.maximum(bottom, torch.tensor(0.01))  # add epsilon.
        dice = top / bottom
        # Compute the Jacc
        top = torch.sum(torch.logical_and(pred_c_i, gt_c_i))
        bottom = torch.sum(torch.logical_or(pred_c_i, gt_c_i))
        bottom = torch.maximum(bottom, torch.tensor(0.01))  # add epsilon.
        jacc = top / bottom

        Dice = Dice + [dice.item()]
        Jacc = Jacc + [jacc.item()]

    # print("Dice:", Dice)
    # print("Jacc:", Jacc)
    return Dice, Jacc


def tre(predictions, labels):
    # pred: warpped fixed points, lab: moving points
    # points tensor shape: (B, N, 3)
    return torch.norm(predictions.float() - labels.float(), p=2, dim=2).mean()


def get_jacobian_determinant(flow):
    e1 = torch.reshape(torch.Tensor([1, 0, 0]), (1, 3, 1, 1, 1)).type(torch.float32).cuda()
    e2 = torch.reshape(torch.Tensor([0, 1, 0]), (1, 3, 1, 1, 1)).type(torch.float32).cuda()
    e3 = torch.reshape(torch.Tensor([0, 0, 1]), (1, 3, 1, 1, 1)).type(torch.float32).cuda()

    stacks = torch.stack([
        flow[..., 1:, :-1, :-1] - flow[..., :-1, :-1, :-1] + e1,
        flow[..., :-1, 1:, :-1] - flow[..., :-1, :-1, :-1] + e2,
        flow[..., :-1, :-1, 1:] - flow[..., :-1, :-1, :-1] + e3
    ], dim=2)
    stacks = torch.permute(stacks, [0, 3, 4, 5, 1, 2])
    determinant = torch.linalg.det(stacks)
    return determinant


def jacobian_det_var(flow):
    determinant = get_jacobian_determinant(flow)
    variance = torch.var(determinant, dim=[1, 2, 3])
    return torch.sqrt(variance)


def neg_jacobian_det(flow):
    determinant = get_jacobian_determinant(flow)
    neg = (determinant <= 0).sum(dim=[1, 2, 3]) / torch.prod(torch.tensor(determinant.size()[1:]))
    return neg * 100


class RecursiveCascadeNetwork(nn.Module):
    def __init__(self, n_cascades, im_size, network):
        super(RecursiveCascadeNetwork, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.det_factor = 0.1
        self.ortho_factor = 0.1
        self.reg_factor = 1.0
        self.gamma = 0.15
        self.lamb = 0.1
        self.beta = 0.1
        self.network = network
        self.n_cascades = n_cascades


        self.stems = []
        # See note in base_networks.py about the assumption in the image shape
        if network == "NIL":
            for i in range(n_cascades):
                self.stems.append(NIL(dim=len(im_size), flow_multiplier=1.0 / n_cascades))
        else:
            assert NotImplementedError(network)

        # Parallelize across all available GPUs
        if torch.cuda.device_count() >= 1:
            self.stems = [nn.DataParallel(model) for model in self.stems]

        for model in self.stems:
            model.to(device)

        self.reconstruction = SpatialTransform(im_size)
        self.reconstruction.to(device)

    def forward(self, fixed, moving, **kwargs):
        stem_results = []  # [{}, {}, ...]

        # Block 0
        block_result = self.stems[0](fixed, moving, flow=kwargs.get("flow", None))
        block_result["warped"] = self.reconstruction(moving, block_result["flow"])
        block_result["agg_flow"] = block_result["flow"]

        # block_result["branch_flow"] = block_result["branch_flow"] * 4.0
        # block_result["branch_warped"] = self.reconstruction(moving, block_result["branch_flow"])
        stem_results.append(block_result)
        # Block i
        for block in self.stems[1:]:
            block_result = block(fixed, stem_results[-1]["warped"], flow=kwargs.get("flow", None))  # keys: ["flow"]
            if len(stem_results) == 1 and 'W' in stem_results[-1]:
                # Block 0 is Affine
                I = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).cuda()
                block_result["agg_flow"] = torch.einsum('bij,bjxyz->bixyz',
                                                        stem_results[-1]['W'] + I,
                                                        block_result['flow']
                                                        ) + stem_results[-1]['flow']
            else:
                # Block 0 is Deform or following Blocks
                block_result["agg_flow"] = self.reconstruction(stem_results[-1]["agg_flow"],
                                                               block_result["flow"]
                                                               ) + block_result["flow"]
            block_result["warped"] = self.reconstruction(moving, block_result["agg_flow"])
            stem_results.append(block_result)

        flows = []
        for res in stem_results:
            flows.append(res["agg_flow"])

        RCN_hyper = {
            "det": self.det_factor,
            "ortho": self.ortho_factor,
            "reg": self.reg_factor
        }
        return flows, stem_results, RCN_hyper


import torch.nn
import torch.nn.functional as F
from timeit import default_timer

from .base_networks import *
from .spatial_transformer import SpatialTransform
from metrics.losses import total_loss_forRCN


def spatial_gradient(flow):
    gradx = (flow[:, :, :-1, :, :] - flow[:, :, 1:, :, :]).unsqueeze(2)
    grady = (flow[:, :, :, :-1, :] - flow[:, :, :, 1:, :]).permute(0, 1, 3, 2, 4).unsqueeze(2)
    gradz = (flow[:, :, :, :, :-1] - flow[:, :, :, :, 1:]).permute(0, 1, 4, 2, 3).unsqueeze(2)
    if gradx.size() == grady.size() == gradz.size():
        return torch.cat([gradx, grady, gradz], dim=2)
    else:
        return [gradx, grady, gradz]


def rmse3D(x, y):
    voxel_wise = torch.sqrt(torch.sum(torch.square(x - y), dim=1))
    return torch.mean(voxel_wise)


def channel_affinity(x):
    B, C, _, _, _ = x.shape
    x = x.view(B, C, -1)
    affinity = torch.bmm(x, x.transpose(1, 2)).view(B, -1)
    return affinity  # (B, ch*ch)


def cosine_sim(x, y):
    # x, y need to be vector
    return torch.mean(torch.nn.functional.cosine_similarity(x, y))


def neighbor_stack(inputTensor, in_chan=3, size=4):
    # stack the tensors in a small window
    I = torch.eye(in_chan * size ** 3).view(in_chan * size ** 3, 3, size, size, size).cuda()
    outputTensor = F.conv3d(inputTensor, I, stride=size)
    return outputTensor


def pair_affinity(x):
    # high computation cost
    B, C, H, W, D = x.shape
    scaler = 16 / H
    if scaler < 1:
        x = nn.Upsample(scale_factor=scaler, mode="trilinear")(x)
    x = x.view(B, C, -1)
    affinity = torch.bmm(x.transpose(1, 2), x).view(B, -1)
    return affinity  # (B, HWD*HWD)


def aggregate_spatial_attention(x, y):
    x1 = torch.mean(torch.square(x), dim=1)
    y1 = torch.mean(torch.square(y), dim=1)
    assert x1.shape == y1.shape
    return torch.mean(torch.square(x1 - y1))


def instance_affinity(x):
    B, C, _, _, _ = x.shape
    x = x.view(B, -1).unsqueeze(0)
    affinity = torch.bmm(x, x.transpose(1, 2))
    return affinity  # (1, B, B)


#####################
### Architectures
#####################


def plain(com, datas, deepSup):
    model, optim, reconstruction = com
    lamb = torch.tensor(0.8)

    rec_loss_mean = []
    model.train()
    while len(datas) > 0:
        fixed, moving = datas.pop()
        flows, results, hyper = model(fixed, moving)
        if deepSup:
            rec_loss, deep_loss_list, affine_loss = total_loss_forRCN(results, hyper, fixed, reconstruction,
                                                                      deep_sup=True)
            for i, l in enumerate(deep_loss_list):
                multiplier = torch.pow(lamb, (len(deep_loss_list) - i)).cuda()
                rec_loss += multiplier * l
            # if affine_loss:
            #     rec_loss += torch.pow(lamb, (len(flows) - 1)) * affine_loss
        else:
            rec_loss = total_loss_forRCN(results, hyper, fixed, reconstruction, deep_sup=False)

        optim.zero_grad()
        rec_loss.backward()
        optim.step()

        rec_loss_mean.append(rec_loss.item())

    rec_loss_mean = np.mean(rec_loss_mean)
    return rec_loss_mean


def KD(S_com, Teacher, datas, hyper):
    Student, S_opt, reconstruction = S_com

    rec_loss_mean = []
    flows_loss_mean = []
    Student.train()
    while len(datas) > 0:
        fixed, moving = datas.pop()
        with torch.no_grad():
            T_flows, T_results, _ = Teacher(fixed, moving)
        S_flows, S_results, S_hyper = Student(fixed, moving)
        rec_loss = total_loss_forRCN(S_results, S_hyper, fixed, reconstruction, deep_sup=False)

        Ldv = torch.tensor(0, dtype=torch.float32).cuda()
        Ldg = torch.tensor(0, dtype=torch.float32).cuda()
        tgrads = spatial_gradient(T_flows[-1])
        sgrads = spatial_gradient(S_flows[0])
        error_v = rmse3D(T_flows[-1], S_flows[0])
        Ldv = Ldv + error_v
        error_g = rmse3D(tgrads, sgrads)
        Ldg = Ldg + error_g

        S_opt.zero_grad()
        (rec_loss + 0.1 * (Ldv + Ldg)).backward()
        S_opt.step()

        rec_loss_mean.append(rec_loss.item())
        flows_loss_mean.append(Ldv.item() + Ldg.item())

    rec_loss_mean = np.mean(rec_loss_mean)
    flows_loss_mean = np.mean(flows_loss_mean)
    return rec_loss_mean, flows_loss_mean



def FitNet(S_com, Teacher, datas, hyper):
    Student, S_opt, reconstruction = S_com
    gamma = hyper["gamma"]

    rec_loss_mean = []
    flows_loss_mean = []
    feats_loss_mean = []
    Student.train()
    MSE = nn.MSELoss()
    while len(datas) > 0:
        fixed, moving = datas.pop()
        with torch.no_grad():
            T_flows, T_results, _ = Teacher(fixed, moving)
        S_flows, S_results, S_hyper = Student(fixed, moving, flow=None)
        rec_loss = total_loss_forRCN(S_results, S_hyper, fixed, reconstruction, deep_sup=False)

        Ldv = torch.tensor(0, dtype=torch.float32).cuda()
        Ldg = torch.tensor(0, dtype=torch.float32).cuda()
        tgrads = spatial_gradient(T_flows[0])
        sgrads = spatial_gradient(S_flows[0])
        error_v = rmse3D(T_flows[0], S_flows[0])
        Ldv = Ldv + error_v

        if type(tgrads) is list:
            error_grad = torch.tensor(0, dtype=torch.float32).cuda()
            for axi in range(3):
                error_grad = error_grad + rmse3D(tgrads[axi], sgrads[axi])
        else:
            error_grad = rmse3D(tgrads, sgrads)
        Ldg = Ldg + error_grad

        feats_loss = torch.tensor(0, dtype=torch.float32).cuda()
        feats_grad_loss = torch.tensor(0, dtype=torch.float32).cuda()
        chwise_loss = torch.tensor(0, dtype=torch.float32).cuda()

        T_feats = T_results[0]["decoder_feas"]
        S_feats = S_results[0]["decoder_regs"]
        for i in range(len(S_feats)):
            if S_feats[i] is None or T_feats[i] is None:
                continue

            feats_loss += rmse3D(S_feats[i], T_feats[i])

            tfeats_grad = spatial_gradient(T_feats[i])
            sfeats_grad = spatial_gradient(S_feats[i])

            if type(tfeats_grad) is list:
                error_grad = torch.tensor(0, dtype=torch.float32).cuda()
                for axi in range(3):
                    error_grad = error_grad + rmse3D(tfeats_grad[axi], sfeats_grad[axi])
            else:
                error_grad = rmse3D(tfeats_grad, sfeats_grad)
            feats_grad_loss += error_grad

            ## channel wise
            ch_affinity_t = channel_affinity(T_feats[i])
            ch_affinity_s = channel_affinity(S_feats[i])
            chwise_loss -= cosine_sim(ch_affinity_s, ch_affinity_t)

        S_opt.zero_grad()

        (rec_loss
         + 0.1 * (Ldv + Ldg)
         + 0.1 * (feats_loss + feats_grad_loss + chwise_loss)
         ).backward()
        S_opt.step()

        # torch.cuda.synchronize()
        rec_loss_mean.append(rec_loss.item())
        flows_loss_mean.append(Ldv.item() + Ldg.item())
        feats_loss_mean.append(feats_loss.item())

    rec_loss_mean = np.mean(rec_loss_mean)
    flows_loss_mean = np.mean(flows_loss_mean)
    feats_loss_mean = np.mean(feats_loss_mean)
    return rec_loss_mean, flows_loss_mean, feats_loss_mean


def twinDistill(com, Teacher, datas, hyper, need_update=True):
    """
    Attention!:
    the self-distilled loss depends on two flows in the stem,
    but we don't want to adjust the deep flow as it is always better than the shallow one.
    """
    model, optim, reconstruction = com
    beta = hyper["beta"]
    lamb = torch.tensor(hyper["lamb"])
    progress = hyper["progress"]  # = x / 1.0

    rec_loss_mean = []
    point_mean = []
    grad_mean = []
    model.train()
    Teacher.eval()
    while len(datas) > 0:
        fixed, moving = datas.pop()
        flows, results, hyper = model(fixed, moving)
        rec_loss, deep_sup_list, affine_loss = total_loss_forRCN(results, hyper, fixed, reconstruction, deep_sup=True)

        pointwise = torch.tensor(0, dtype=torch.float32).cuda()
        gradwise = torch.tensor(0, dtype=torch.float32).cuda()
        deepsup = torch.tensor(0, dtype=torch.float32).cuda()

        with torch.no_grad():
            T_flows, _, _ = Teacher(fixed, moving)

        # Dense instruction
        for i in range(0, len(flows) - 1):
            # w/ same level: len(flows), w/o same level len(flows)-1
            # w/ 1st Block: begin:0, w/o 1st Block: begin:1
            guided = flows[i]
            guided_grads = spatial_gradient(guided)
            point_tmp = torch.tensor(0, dtype=torch.float32).cuda()
            grad_tmp = torch.tensor(0, dtype=torch.float32).cuda()
            for j in range(i + 1, len(flows)):
                # w/ same level: i, w/o same level: i+1
                instructor = T_flows[j]
                instructor_grads = spatial_gradient(instructor)
                multiplier = torch.pow(lamb, (j - i))
                error_point = rmse3D(instructor.detach(), guided)
                error_grad = rmse3D(instructor_grads.detach(), guided_grads)
                point_tmp += multiplier * error_point
                grad_tmp += multiplier * error_grad
            pointwise += point_tmp  # / (len(flows) - i)
            gradwise += grad_tmp  # / (len(flows) - i)

        # final Instruction
        # instructor = T_flows[-1]
        # instructor_grads = spatial_gradient(instructor)
        # for i in range(1, len(flows)):
        #     # w/ same level: len(flows), w/o same level len(flows)-1
        #     guided = flows[i]
        #     guided_grads = spatial_gradient(guided)
        #     multiplier = torch.pow(lamb, (len(flows) - i - 1))
        #     pointwise += multiplier * rmse3D(instructor.detach(), guided)
        #     gradwise += multiplier * rmse3D(instructor_grads.detach(), guided_grads)

        # deepsup
        # why need to supervise the middle D block, instead of only D1
        for i, l in enumerate(deep_sup_list):
            multiplier = torch.pow(lamb, (len(deep_sup_list) - i)).cuda()
            deepsup += multiplier * l

        rec_loss = rec_loss + deepsup
        # if affine_loss:
        #     rec_loss = rec_loss + torch.pow(lamb, (len(flows) - 1)) * affine_loss

        if need_update:
            optim.zero_grad()
            (rec_loss + beta * progress * (pointwise + gradwise) + deepsup).backward()
            # (beta * (pointwise + gradwise)).backward()
            optim.step()

        rec_loss_mean.append(rec_loss.item())
        point_mean.append(pointwise.item())
        grad_mean.append(gradwise.item())

    rec_loss_mean = np.mean(rec_loss_mean)
    point_mean = np.mean(point_mean)
    grad_mean = np.mean(grad_mean)
    return rec_loss_mean, point_mean, grad_mean


def selfDistill(com, Teacher, datas, hyper, need_update=True):
    """
    Attention!:
    the self-distilled loss depends on two flows in the stem,
    but we don't want to adjust the deep flow as it is always better than the shallow one.
    """
    model, optim, reconstruction = com
    progress = hyper["progress"]  # = x / 1.0
    lamb = 0.8
    guided_list = [0]

    rec_loss_mean = []
    point_mean = []
    grad_mean = []
    model.train()
    Teacher.eval()
    while len(datas) > 0:
        fixed, moving = datas.pop()

        flows, results, hyper = model(fixed, moving)
        rec_loss_raw, rec_loss_reg, deep_sup_list, affine_loss = total_loss_forRCN(results, hyper, fixed,
                                                                                   reconstruction,
                                                                                   deep_sup=True, sep_reg=True)

        pointwise = torch.tensor(0, dtype=torch.float32).cuda()
        gradwise = torch.tensor(0, dtype=torch.float32).cuda()
        deepsup = torch.tensor(0, dtype=torch.float32).cuda()

        with torch.no_grad():
            T_flows, _, _ = Teacher(fixed, moving)

        instructor = T_flows[-1]
        instructor_grads = spatial_gradient(instructor)
        # D_n instruct D_i
        if len(guided_list) > 0:
            for i in guided_list:
                # w/ same level: len(flows), w/o same level len(flows)-1
                # w/ 1st Block: begin:0, w/o 1st Block: begin:1
                guided = flows[i]
                guided_grads = spatial_gradient(guided)

                if type(instructor_grads) is list:
                    error_point = rmse3D(instructor.detach(), guided)
                    error_grad = torch.tensor(0, dtype=torch.float32).cuda()
                    for axi in range(3):
                        error_grad = error_grad + rmse3D(instructor_grads[axi].detach(), guided_grads[axi])
                else:
                    error_point = rmse3D(instructor.detach(), guided)
                    error_grad = rmse3D(instructor_grads.detach(), guided_grads)

                pointwise += error_point * (lamb ** (len(flows) - i - 1))
                gradwise += error_grad * (lamb ** (len(flows) - i - 1))
                deepsup += deep_sup_list[i] * (lamb ** (len(flows) - i - 1))

        if need_update:
            optim.zero_grad()
            (rec_loss_raw + rec_loss_reg
             + progress * (0.4 * (pointwise + gradwise)) + progress * (4.0 * deepsup)
             ).backward()
            optim.step()

        rec_loss_mean.append((rec_loss_raw + rec_loss_reg).item())
        point_mean.append(pointwise.item())
        grad_mean.append(gradwise.item())

    rec_loss_mean = np.mean(rec_loss_mean)
    point_mean = np.mean(point_mean)
    grad_mean = np.mean(grad_mean)

    return rec_loss_mean, point_mean, grad_mean

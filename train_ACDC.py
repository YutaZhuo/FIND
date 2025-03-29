import argparse
import os
import torch
from timeit import default_timer
import time
from networks.recursive_cascade_networks import RecursiveCascadeNetwork, mask_metrics_ACDC, \
    jacobian_det_var, neg_jacobian_det
from networks.architecture import *

from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from metrics.losses import total_loss
import data_util as Data
from math import *
import numpy as np

dataset_opt = {"dataroot": "../datasets/ACDC/training",
               "batch_size": 4,
               "use_shuffle": True,
               "num_workers": 8,
               "finesize": [64, 128, 128]}


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.name = args.name + time.strftime("%m%d%H%M%S")

    if not os.path.exists('./logs/{}/model_wts'.format(args.name)):
        os.makedirs('./logs/{}/model_wts'.format(args.name))
    if not os.path.exists('./logs/{}/tensorboard'.format(args.name)):
        os.makedirs('./logs/{}/tensorboard'.format(args.name))

    writer = SummaryWriter(log_dir="./logs/{}/tensorboard".format(args.name))

    print(args)

    dataset = Data.create_dataset_ACDC(dataroot=dataset_opt["dataroot"],
                                       finesize=dataset_opt["finesize"],
                                       phase="train")
    train_loader = Data.create_dataloader(dataset, dataset_opt, "train")
    training_iters = int(ceil(dataset.data_len / float(dataset_opt["batch_size"]))) * args.epochs
    reconstruction = SpatialTransform(dataset_opt["finesize"]).cuda()

    model = RecursiveCascadeNetwork(n_cascades=args.n_cascades,
                                    im_size=dataset_opt["finesize"],
                                    network=args.network).cuda()

    Teacher = None
    if args.strategy == "plain":
        trainable_params = []
        for submodel in model.stems:
            trainable_params += list(submodel.parameters())

        optim = Adam(trainable_params, lr=1e-4)
        scheduler = MultiStepLR(optimizer=optim, milestones=[training_iters * 0.6, training_iters * 0.8], gamma=0.5)
    elif args.strategy == "Fitnet":
        Teacher = RecursiveCascadeNetwork(n_cascades=4,
                                          im_size=dataset_opt["finesize"],
                                          network="VTN").cuda()
        params_dict = torch.load("")
        Teacher.eval()
        for i, submodel in enumerate(Teacher.stems):
            submodel.load_state_dict(params_dict["cascade {}".format(i)])

        trainable_params = []
        for submodel in model.stems:
            trainable_params += list(submodel.parameters())

        optim = Adam(trainable_params, lr=1e-4)
        scheduler = MultiStepLR(optimizer=optim, milestones=[training_iters * 0.6, training_iters * 0.8], gamma=0.5)
    elif args.strategy == "selfdistill":
        trainable_params = []
        for submodel in model.stems:
            trainable_params += list(submodel.parameters())
        optim = Adam(trainable_params, lr=1e-4)
        scheduler = MultiStepLR(optimizer=optim, milestones=[training_iters * 0.6, training_iters * 0.8], gamma=0.5)
        # aux model don't need training
        model_aux = RecursiveCascadeNetwork(n_cascades=args.n_cascades,
                                            im_size=dataset_opt["finesize"],
                                            network=args.network).cuda()

    else:
        raise NotImplementedError(args.strategy)

    # Saving the losses
    record_cnter = 0
    for curr_epoch in range(1, args.epochs + 1):
        for istep, train_data in enumerate(train_loader, start=1):
            print(f"-----Iteration {record_cnter} / {training_iters}-----")
            print(f">>>>> Train:")
            datas = [(train_data["F"].cuda(), train_data["M"].cuda())]
            assert list(datas[0][0].shape[2:]) == dataset_opt["finesize"], "data shape {} error".format(
                list(datas[0][0].shape[2:]))
            t0 = default_timer()

            if args.strategy == "plain":
                print("LR:", scheduler.get_last_lr())

                component = (model, optim, reconstruction)
                rec_loss_mean = plain(component, datas, deepSup=False)
                if record_cnter % args.record_freq == 0:
                    writer.add_scalar(tag="Loss/reconstruction",
                                      scalar_value=rec_loss_mean,
                                      global_step=record_cnter)
                print("Loss: {}".format(round(rec_loss_mean, 4)))
                scheduler.step()
            elif args.strategy in "Fitnet":
                print("LR:", scheduler.get_last_lr())
                S_com = (model, optim, reconstruction)
                hyper = {
                    "gamma": 0.8,
                }
                rec_loss_mean, flows_loss_mean, feats_loss_mean = FitNet(S_com, Teacher, datas, hyper)
                if record_cnter % args.record_freq == 0:
                    writer.add_scalar(tag="Loss/reconstruction",
                                      scalar_value=rec_loss_mean,
                                      global_step=record_cnter)
                    writer.add_scalar(tag="Loss/flow_dist",
                                      scalar_value=flows_loss_mean,
                                      global_step=record_cnter)
                    writer.add_scalar(tag="Loss/feature_dist",
                                      scalar_value=feats_loss_mean,
                                      global_step=record_cnter)
                print(
                    "Rec loss: {}, flow loss: {}, feats loss: {}".format(round(rec_loss_mean, 4),
                                                                         round(flows_loss_mean, 4),
                                                                         round(feats_loss_mean, 4)))
                scheduler.step()

            elif args.strategy == "selfdistill":
                print("LR:", scheduler.get_last_lr())
                hyper = {
                    "progress": 0.0,
                }
                big_step = 1000
                stage_num = training_iters // big_step
                stage = record_cnter // big_step
                hyper["progress"] = stage * 1.0 / stage_num  # set progress as float
                print("stage {}: progress={}".format(stage, hyper["progress"]))

                component = (model, optim, reconstruction)
                rec_loss_mean, point_mean, grad_mean = selfDistill(component, model_aux, datas, hyper)

                writer.add_scalar(tag="Loss/reconstruction",
                                  scalar_value=rec_loss_mean,
                                  global_step=record_cnter)
                writer.add_scalar(tag="Loss/point",
                                  scalar_value=point_mean,
                                  global_step=record_cnter)
                writer.add_scalar(tag="Loss/grad",
                                  scalar_value=grad_mean,
                                  global_step=record_cnter)

                print("Rec loss: {}, point: {}, grad: {}".format(round(rec_loss_mean, 4),
                                                                 round(point_mean, 4),
                                                                 round(grad_mean, 4)))
                scheduler.step()

                if record_cnter % big_step == 0:
                    print("Update the aux model")
                    # update the model_aux
                    params_new = {}
                    for i, submodel in enumerate(model.stems):
                        params_new[f"cascade {i}"] = submodel.state_dict()
                    for i, submodel in enumerate(model_aux.stems):
                        submodel.load_state_dict(params_new["cascade {}".format(i)])

            else:
                raise NotImplementedError(args.strategy)
            record_cnter = record_cnter + 1

            t1 = default_timer()
            print("train time: {}".format(t1 - t0))

            if record_cnter == 1 or record_cnter % args.val_freq == 0:
                print(">>>>> Validation:")
                model.eval()

                val_epoch_loss = []
                val_epoch_dsc = []
                val_epoch_jacc = []
                val_epoch_jacobi_var = []
                val_epoch_neg_jacobi = []

                with torch.no_grad():
                    dataset = Data.create_dataset_ACDC(dataroot="../datasets/ACDC/training",
                                                       finesize=dataset_opt["finesize"],
                                                       phase="val")
                    val_loader = Data.create_dataloader(dataset, dataset_opt, "val")
                    for idx, val_data in enumerate(val_loader):
                        fixed, moving = val_data["F"].cuda(), val_data["M"].cuda()
                        fixed_seg, moving_seg = val_data["FS"].cuda(), val_data["MS"].cuda()

                        flows, results, hyper = model(fixed, moving)
                        loss = total_loss_forRCN(results, hyper, fixed, reconstruction)
                        val_epoch_loss.append(loss.item())
                        dsc_blocks = []
                        jacc_blocks = []
                        jacobi_var_blocks = []
                        neg_jacobi_blocks = []
                        for flow in flows:
                            warped_seg = reconstruction(moving_seg, flow)
                            dice_scores, jacc_scores = mask_metrics_ACDC(fixed_seg, warped_seg)
                            jacobi_det_var = jacobian_det_var(flow)
                            neg_jacobi = neg_jacobian_det(flow)
                            dsc_blocks.append(dice_scores)
                            jacc_blocks.append(jacc_scores)
                            jacobi_var_blocks.append(jacobi_det_var.item())
                            neg_jacobi_blocks.append(neg_jacobi.item())

                        val_epoch_dsc.append(dsc_blocks)
                        val_epoch_jacc.append(jacc_blocks)
                        val_epoch_jacobi_var.append(jacobi_var_blocks)
                        val_epoch_neg_jacobi.append(neg_jacobi_blocks)

                val_epoch_dsc = torch.tensor(val_epoch_dsc)
                val_epoch_jacc = torch.tensor(val_epoch_jacc)
                val_loss = np.mean(val_epoch_loss)
                val_dsc = torch.mean(val_epoch_dsc, dim=[0, 2]).numpy()
                val_jacc = torch.mean(val_epoch_jacc, dim=[0, 2]).numpy()
                val_jacobi_var = np.mean(val_epoch_jacobi_var, axis=0)
                val_NJD = np.mean(val_epoch_neg_jacobi, axis=0)
                writer.add_scalar(tag="Loss/Valid loss",
                                  scalar_value=val_loss,
                                  global_step=record_cnter)
                for i, value in enumerate(val_dsc):
                    writer.add_scalar(tag="Dice/block{}".format(i),
                                      scalar_value=value,
                                      global_step=record_cnter)
                for i, value in enumerate(val_jacc):
                    writer.add_scalar(tag="JACC/block{}".format(i),
                                      scalar_value=value,
                                      global_step=record_cnter)
                for i, value in enumerate(val_jacobi_var):
                    writer.add_scalar(tag="JacobiVar/block{}".format(i),
                                      scalar_value=value,
                                      global_step=record_cnter)
                for i, value in enumerate(val_NJD):
                    writer.add_scalar(tag="NJD/block{}".format(i),
                                      scalar_value=value,
                                      global_step=record_cnter)
                print("DSC: {}".format(np.mean(val_epoch_dsc, axis=0)))
                print("JACC: {}".format(np.mean(val_epoch_jacc, axis=0)))

            if record_cnter % args.ckp_freq == 0 or record_cnter == training_iters:
                ckp = {}
                for i, submodel in enumerate(model.stems):
                    ckp[f"cascade {i}"] = submodel.state_dict()
                torch.save(ckp, "./logs/{}/model_wts/E{}I{}.pth".format(args.name, curr_epoch, istep))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', "--n_cascades", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--ckp_freq", type=int, default=10000)
    parser.add_argument("--val_freq", type=int, default=10000)
    parser.add_argument("--record_freq", type=int, default=100)

    parser.add_argument('-g', "--gpu", type=str)
    parser.add_argument('--network', type=str)
    parser.add_argument("--aldk", action="store_true")

    parser.add_argument("--strategy", type=str,
                        help="//FitNet_Nice//plain//selfdistill//")
    parser.add_argument("--finetune", action="store_true",
                        help="may used for Early stop distillation")
    parser.add_argument("--pretrained_model", type=str,
                        help="load the model weight from dir")
    parser.add_argument("--name", type=str, default="debug")


    args = parser.parse_args()

    main(args)

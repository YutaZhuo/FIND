"""
This is a loader for our specific purpouse, please implement your own loader.
"""
import os
import torch
import numpy as np
import json
import h5py
import _pickle as pickle
from torch.nn import functional as F
from networks.spatial_transformer import SpatialTransform, warp3D
from monai.transforms import RandAffined, Rand3DElasticd, RandRotate90d, RandFlipd, Compose, apply_transform


def sample_power(lo, hi, k, size):
    size = size.type(torch.int16).tolist()
    r = (hi - lo) / 2
    center = (hi + lo) / 2
    r = r ** (1 / k)
    points = (torch.rand(size, dtype=torch.float32) - 0.5) * 2 * r
    points = (torch.abs(points) ** k) * torch.sign(points)
    return points + center


def get_coef(u):
    return torch.stack([((1 - u) ** 3) / 6, (3 * (u ** 3) - 6 * (u ** 2) + 4) / 6,
                        (-3 * (u ** 3) + 3 * (u ** 2) + 3 * u + 1) / 6, (u ** 3) / 6], dim=1)


def free_form_fields(shape, control_fields):
    interp_range = 4
    control_fields = torch.Tensor(control_fields).type(torch.float32).cuda()
    _, _, n, m, t = list(control_fields.shape)
    # "same" padding
    control_fields = F.pad(control_fields, [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], "constant")
    control_fields = torch.reshape(torch.permute(control_fields, (2, 3, 4, 0, 1)), (n + 2, m + 2, t + 2, -1))

    assert shape[0] % (n - 1) == 0
    s_x = shape[0] // (n - 1)
    u_x = (torch.arange(0, s_x, dtype=torch.float32) + 0.5) / s_x  # s_x
    coef_x = get_coef(u_x).cuda()  # (s_x, 4)
    shape_cf = list(control_fields.shape)  # (n+2, m+2, t+2, Bs*C)
    flow = torch.cat([torch.matmul(coef_x, torch.reshape(control_fields[i: i + interp_range], [interp_range, -1]))
                      for i in range(0, n - 1)], dim=0)

    assert shape[1] % (m - 1) == 0
    s_y = shape[1] // (m - 1)
    u_y = (torch.arange(0, s_y, dtype=torch.float32) + 0.5) / s_y
    coef_y = get_coef(u_y).cuda()  # (s_y, 4)
    dimseq = list(range(len(flow.shape) - 1, -1, -1))
    flow = torch.reshape(torch.permute(flow, dims=dimseq), [shape_cf[1], -1])
    flow = torch.cat([torch.matmul(coef_y, torch.reshape(flow[i:i + interp_range], [interp_range, -1]))
                      for i in range(0, m - 1)], dim=0)

    assert shape[2] % (t - 1) == 0
    s_z = shape[2] // (t - 1)
    u_z = (torch.arange(0, s_z, dtype=torch.float32) + 0.5) / s_z
    coef_z = get_coef(u_z).cuda()  # (s_z, 4)
    dimseq = list(range(len(flow.shape) - 1, -1, -1))
    flow = torch.reshape(torch.permute(flow, dims=dimseq), [shape_cf[2], -1])
    flow = torch.cat([torch.matmul(coef_z, torch.reshape(flow[i:i + interp_range], [interp_range, -1]))
                      for i in range(0, t - 1)], dim=0)

    flow = torch.reshape(flow, [shape[2], -1, 3, shape[1], shape[0]])  # reshape seems strange
    flow = torch.permute(flow, [1, 2, 4, 3, 0])
    return flow


def meshgrids(shape, flatten=True):
    indices_x = torch.arange(0, shape[2])
    indices_y = torch.arange(0, shape[3])
    indices_z = torch.arange(0, shape[4])
    indices = torch.stack(
        torch.meshgrid(indices_x, indices_y, indices_z, indexing="ij"),
        dim=0)
    indices = torch.tile(
        torch.unsqueeze(indices, dim=0),
        [shape[0], 1, 1, 1, 1]
    ).type(torch.float32)
    if flatten:
        return torch.reshape(
            indices,
            torch.stack([shape[0], 3, shape[2] * shape[3] * shape[4]])
        )
    else:
        return indices


def meshgrids_like(tensor, flatten=True):
    return meshgrids(tensor.shape, flatten)


def warp_points(flow, pts):
    # flow: (B, 3, X, Y, Z), pts: (B, 3, 6)
    moving_pts = meshgrids_like(flow, flatten=False).cuda() + flow
    shape = flow.shape
    moving_pts = torch.reshape(
        moving_pts,
        [shape[0], 3, shape[2] * shape[3] * shape[4], 1]
    )
    distance = torch.sqrt(
        torch.sum((moving_pts - torch.unsqueeze(pts, dim=2)) ** 2, dim=1)
    )
    closet = torch.argmin(distance, dim=1).type(torch.int32)  # (B, 6)
    fixed_pts = torch.stack([
        closet // (shape[3] * shape[4]),
        (closet // shape[4]) % shape[3],
        closet % shape[4]
    ], dim=1)
    return fixed_pts


def get_data(ret, reconstruction, is_train):
    divisor = torch.tensor([255.]).cuda()
    fixed = torch.from_numpy(ret["voxel1"]).cuda() / divisor
    moving = torch.from_numpy(ret["voxel2"]).cuda() / divisor
    fixed_seg = (torch.from_numpy(ret["seg1"]).cuda() > 128).type(torch.float32)
    moving_seg = (torch.from_numpy(ret["seg2"]).cuda() > 128).type(torch.float32)
    fixed_point = torch.from_numpy(ret["point1"]).cuda()
    moving_point = torch.from_numpy(ret["point2"]).cuda()

    def augmentation(x, size):
        elastic_transform = Rand3DElasticd(
            keys=['moving_img', 'moving_seg'],  # 字典中图像和标签的键名
            mode=('bilinear', 'nearest'),  # 分别为图像和mask设置插值方式
            prob=1.0,  # 变换应用的概率
            sigma_range=(5, 7),  # 高斯核的标准差范围
            magnitude_range=(80, 120),  # 弹性变形的幅度范围
            spatial_size=size  # 图像的空间大小
        )
        compose_transform = Compose([
            RandRotate90d(
                keys=['moving_img', 'moving_seg', 'fixed_img', 'fixed_seg'],
                prob=0.5,
                spatial_axes=(1, 2)
            ),
            RandFlipd(
                keys=['moving_img', 'moving_seg', 'fixed_img', 'fixed_seg'],
                prob=0.5,
                spatial_axis=1
            ),
            RandFlipd(
                keys=['moving_img', 'moving_seg', 'fixed_img', 'fixed_seg'],
                prob=0.5,
                spatial_axis=2
            )
        ])
        aug_data = elastic_transform({"moving_img": x["moving_img"],
                                      "moving_seg": x["moving_seg"]})
        aug_data = compose_transform({"moving_img": aug_data["moving_img"],
                                      "moving_seg": aug_data["moving_seg"],
                                      "fixed_img": x["fixed_img"],
                                      "fixed_seg": x["fixed_seg"]})
        return aug_data

    if is_train:
        for b in range(fixed.shape[0]):
            aug_data = augmentation({"moving_img": moving[b],
                                     "moving_seg": moving_seg[b],
                                     "fixed_img": fixed[b],
                                     "fixed_seg": fixed_seg[b]
                                     }, size=fixed.shape[2:])
            moving[b] = aug_data["moving_img"]
            moving_seg[b] = aug_data["moving_seg"]
            fixed[b] = aug_data["fixed_img"]
            fixed_seg[b] = aug_data["fixed_seg"]

    return fixed, moving, fixed_seg, moving_seg, fixed_point, None


class Hdf5Reader:
    def __init__(self, path):
        try:
            self.file = h5py.File(path, "r")
        except Exception:
            print('{} not found!'.format(path))
            self.file = None

    def __getitem__(self, key):
        data = {'id': key}
        if self.file is None:
            return data
        group = self.file[key]
        for k in group:
            data[k] = group[k]
        return data


class FileManager:
    def __init__(self, files):
        self.files = {}
        for k, v in files.items():
            self.files[k] = Hdf5Reader(v["path"])

    def __getitem__(self, key):
        p = key.find('/')
        if key[:p] in self.files:
            ret = self.files[key[:p]][key[p + 1:]]
            ret['id'] = key.replace('/', '_')
            return ret
        elif '/' in self.files:
            ret = self.files['/'][key]
            ret['id'] = key.replace('/', '_')
            return ret
        else:
            raise KeyError('{} not found'.format(key))


class Dataset:
    def __init__(self, split_path, image_size=128, affine=False,
                 mask=False, paired=False, task=None,
                 batch_size=4):

        with open(split_path, 'r') as f:
            config = json.load(f)
        self.files = FileManager(config['files'])  # self.files is a dict: {filename: data}
        self.subset = {}
        self.fraction = image_size * 1.0 / 128
        self.image_size = image_size

        for k, v in config['subsets'].items():
            self.subset[k] = {}
            print(k)
            for entry in v:
                self.subset[k][entry] = self.files[entry]
        # self.subset is a dict: {subset_name: {filename: data}}

        self.paired = paired

        def convert_int(key):
            try:
                return int(key)
            except ValueError as e:
                return key

        self.schemes = dict([(convert_int(k), v)
                             for k, v in config['schemes'].items()])

        for k, v in self.subset.items():
            print('Number of data in {} is {}'.format(k, len(v)))

        self.task = task
        if self.task is None:
            self.task = config.get("task", "registration")
        if not isinstance(self.task, list):
            self.task = [self.task]

        self.batch_size = batch_size

    # //__init__

    def get_pairs_adj(self, data):
        pairs = []
        d1 = None
        for d2 in data:
            if d1 is None:
                d1 = d2
            else:
                pairs.append((d1, d2))
                pairs.append((d2, d1))
                d1 = None
        return pairs

    def get_pairs(self, data, ordered=True):
        pairs = []
        for i, d1 in enumerate(data):
            for j, d2 in enumerate(data):
                if i != j:
                    if ordered or i < j:
                        pairs.append((d1, d2))
        return pairs

    def generate_pairs(self, arr, loop=False):
        # generate_pairs is a generator
        if self.paired:
            sets = self.get_pairs_adj(arr)
        else:
            sets = self.get_pairs(arr, ordered=True)

        while True:
            if loop:
                np.random.shuffle(sets)
            for d1, d2 in sets:
                yield (d1, d2)
            if not loop:
                break

    def generator(self, subset, batch_size=None, loop=False, aldk=False):
        # using yield command, func generator becomes a generator
        if batch_size is None:
            batch_size = self.batch_size
        valid_mask = np.ones([6], dtype=np.bool)
        scheme = self.schemes[subset]  # scheme is a dict: {subset_name: fraction}
        num_data = 0
        if 'registration' in self.task:
            generators = [(self.generate_pairs(list(self.subset[k].values()), loop))
                          for k, fraction in scheme.items()]  # give datas to generate data-pairs to registration
            fractions = [int(np.round(fraction * batch_size))
                         for k, fraction in scheme.items()]

            while True:
                ret = dict()
                ret['voxel1'] = np.zeros(
                    (batch_size, 1, self.image_size, self.image_size,
                     self.image_size), dtype=np.float32)
                ret['voxel2'] = np.zeros(
                    (batch_size, 1, self.image_size, self.image_size,
                     self.image_size), dtype=np.float32)
                ret['seg1'] = np.zeros(
                    (batch_size, 1, self.image_size, self.image_size,
                     self.image_size), dtype=np.float32)
                ret['seg2'] = np.zeros(
                    (batch_size, 1, self.image_size, self.image_size,
                     self.image_size), dtype=np.float32)
                ret['point1'] = np.ones(
                    (batch_size, 3, np.sum(valid_mask)), dtype=np.float32) * (-1)
                ret['point2'] = np.ones(
                    (batch_size, 3, np.sum(valid_mask)), dtype=np.float32) * (-1)
                ret['agg_flow'] = np.zeros(
                    (batch_size, 3, 128, 128, 128), dtype=np.float32)
                ret['id1'] = np.empty((batch_size), dtype='<U40')
                ret['id2'] = np.empty((batch_size), dtype='<U40')

                i = 0
                flag = True
                nums = fractions
                for gen, num in zip(generators, nums):
                    assert not self.paired or num % 2 == 0
                    for t in range(num):
                        try:
                            while True:
                                d1, d2 = next(gen)
                                break
                        except StopIteration:
                            flag = False
                            break

                        ret['voxel1'][i, 0, ...], ret['voxel2'][i, 0, ...] \
                            = d1['volume'], d2['volume']

                        if 'segmentation' in d1:
                            ret['seg1'][i, 0, ...] = d1['segmentation']
                        if 'segmentation' in d2:
                            ret['seg2'][i, 0, ...] = d2['segmentation']
                        # if 'point' in d1:
                        #     ret['point1'][i] = d1['point'][...][valid_mask]
                        # if 'point' in d2:
                        #     ret['point2'][i] = d2['point'][...][valid_mask]

                        ret['id1'][i] = d1['id']
                        ret['id2'][i] = d2['id']
                        i += 1
                num_data += 1
                if flag:
                    assert i == batch_size
                    yield ret
                else:
                    yield ret
                    break


def prepare_data(iters, generator, reconstruction, is_train, image_size):
    datas = []
    for _ in range(iters):
        ret = next(generator)
        # print(ret["id1"], ret["id2"])
        fixed, moving, _, _, _, _ = get_data(ret, reconstruction, is_train=is_train)
        datas.append([fixed, moving])
    return datas

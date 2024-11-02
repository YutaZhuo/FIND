import os
import torch
import random
import numpy as np
from monai.transforms import Rand3DElasticd, RandRotate90d, RandFlipd, Compose, apply_transform

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(x, size, split='val'):
    if split == "train":
        compose_transform = Compose([
            # Rand3DElasticd(
            #     keys=['moving_img', 'moving_seg'],  # 字典中图像和标签的键名
            #     mode=('bilinear', 'nearest'),  # 分别为图像和mask设置插值方式
            #     prob=1.0,  # 变换应用的概率
            #     sigma_range=(5, 7),  # 高斯核的标准差范围
            #     magnitude_range=(80, 120),  # 弹性变形的幅度范围
            #     spatial_size=size  # 图像的空间大小
            # ),
            # RandRotate90d(
            #     keys=['moving_img', 'moving_seg', 'fixed_img', 'fixed_seg'],
            #     prob=0.5,
            #     # spatial_axes=(1, 2)  # ACDC, Brain, Liver
            #     spatial_axes=(0, 2)  # NLST
            # ),
            # RandFlipd(
            #     keys=['moving_img', 'moving_seg', 'fixed_img', 'fixed_seg'],
            #     prob=0.5,
            #     spatial_axis=1
            # ),
            # RandFlipd(
            #     keys=['moving_img', 'moving_seg', 'fixed_img', 'fixed_seg'],
            #     prob=0.5,
            #     spatial_axis=2
            # ),
            # RandFlipd(
            #     keys=['moving_img', 'moving_seg', 'fixed_img', 'fixed_seg'],
            #     prob=0.5,
            #     spatial_axis=0
            # )
        ])
        aug_data = compose_transform({"moving_img": x["moving_img"],
                                      "moving_seg": x["moving_seg"],
                                      "fixed_img": x["fixed_img"],
                                      "fixed_seg": x["fixed_seg"]})
        return [aug_data["fixed_img"],
                aug_data["moving_img"],
                aug_data["fixed_seg"],
                aug_data["moving_seg"]]
    else:
        return [x["fixed_img"],
                x["moving_img"],
                x["fixed_seg"],
                x["moving_seg"]]


def transform_augment(img_list, split='val', min_max=(0, 1)):
    ret_img = []
    for img in img_list[:-2]:
        img = torch.from_numpy(img).float().unsqueeze(0)
        img = img * (min_max[1] - min_max[0]) + min_max[0]
        ret_img.append(img)
        # print(img.shape)
    for img in img_list[-2:]:
        img = torch.from_numpy(img).float().unsqueeze(0)
        ret_img.append(img)
    ret_img = augment({"fixed_img": ret_img[0],
                       "moving_img": ret_img[1],
                       "fixed_seg": ret_img[2],
                       "moving_seg": ret_img[3]},
                      size=img_list[0].shape, split=split)

    return ret_img

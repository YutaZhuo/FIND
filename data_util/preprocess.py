import SimpleITK as sitk
import os
import shutil
from monai import transforms
import numpy as np
import json
import pandas as pd

import matplotlib.pyplot as plt

"""
只包含数据路径调整, 数据集划分, 图像Crop和Resize. 不包含数据归一化!
确保各向同性, 数据归一化范围[0, 1]
"""


def ACDC():
    """
    voxel space = 1.5
    target size = [64, 128, 128]
    """
    source = "../../myDiff/ACDC/database/testing"
    target = "../datasets/ACDC/testing"
    files = ["fixed.nii.gz", "fixed_gt.nii.gz", "moving.nii.gz", "moving_gt.nii.gz"]

    for folder in os.listdir(source):
        if os.path.isdir(os.path.join(source, folder)):
            source_folder = os.path.join(source, folder)
            target_folder = os.path.join(target, folder)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder, exist_ok=True)
            else:
                pass
            for f in files:
                source_file = os.path.join(source_folder, f)
                img = sitk.ReadImage(source_file)
                space = img.GetSpacing()
                dim = img.GetSize()
                new_space = (1.5, 1.5, 1.5)
                new_size = [int(round(osz * ospc / tspc)) for osz, ospc, tspc in
                            zip(dim, space, new_space)]
                if "gt" in f:
                    interpolater = sitk.sitkNearestNeighbor
                else:
                    interpolater = sitk.sitkBSpline
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(img)
                resampler.SetOutputSpacing(new_space)
                resampler.SetInterpolator(interpolater)
                resampler.SetSize(new_size)
                resampled_img = resampler.Execute(img)

                # :[64, 128, 128]
                resampled_data = sitk.GetArrayFromImage(resampled_img)
                resampled_data = resampled_data[None, :]
                if "gt" in f:
                    trans = transforms.Compose([
                        transforms.ResizeWithPadOrCrop(spatial_size=[64, 128, 128], mode="constant"),
                    ])
                else:
                    trans = transforms.Compose([
                        transforms.ScaleIntensity(),
                        transforms.ResizeWithPadOrCrop(spatial_size=[64, 128, 128], mode="constant")
                    ])
                trans_data = trans(resampled_data)
                trans_img = sitk.GetImageFromArray(trans_data[0])
                trans_img.SetSpacing([1.5, 1.5, 1.5])
                sitk.WriteImage(trans_img, os.path.join(target_folder, f))


def OASIS():
    phase = "testing"
    path = "../datasets/OASIS/{}/images".format(phase)
    file_json = []
    for f in os.listdir(path):
        for m in os.listdir(path):
            if f != m and np.random.rand() < 0.01:
                cfg_dict = {}
                cfg_dict["images_fixed"] = os.path.join("images", f).replace("\\", "/")
                cfg_dict["images_moving"] = os.path.join("images", m).replace("\\", "/")
                cfg_dict["labels_fixed"] = os.path.join("labels", f).replace("\\", "/")
                cfg_dict["labels_moving"] = os.path.join("labels", m).replace("\\", "/")

                file_json.append(cfg_dict)
    print(len(file_json))
    with open(os.path.join("../datasets/OASIS/{}".format(phase), "val.json"), "w", encoding="utf-8") as f:
        json.dump(file_json, f, indent=4, separators=(",", ":"), ensure_ascii=False)


def Brain():
    """
    {
        "image_fixed":"img/liver_132.nii.gz",
        "image_moving":"img/liver_133.nii.gz",
        "label_fixed":"mask/liver_132_mask.nii.gz",
        "label_moving":"mask/liver_133_mask.nii.gz"
    },
    """
    path = "../datasets/lpba/img"
    # prefix_list = {"abidef": [], "abide": [], "adhd": [], "adni": []}
    prefix_list = {"lpba": []}

    file_json = []
    for file in os.listdir(path):
        prefix = file.split("_")[0]
        prefix_list[prefix].append(file)

    for k, v in prefix_list.items():
        print(k)
        for f1 in v:
            for f2 in v:
                # if f1 != f2 and (
                #         (k == "abidef" or k == "adhd") and np.random.rand() < 0.15) or (k != "abidef" and k != "adhd"):
                if f1 != f2:
                    cfg_dict = {}
                    cfg_dict["image_fixed"] = os.path.join("img", f1).replace("\\", "/")
                    cfg_dict["image_moving"] = os.path.join("img", f2).replace("\\", "/")
                    m1 = f1.split(".")[0] + "_mask.nii.gz"
                    m2 = f2.split(".")[0] + "_mask.nii.gz"
                    cfg_dict["label_fixed"] = os.path.join("mask", m1).replace("\\", "/")
                    cfg_dict["label_moving"] = os.path.join("mask", m2).replace("\\", "/")
                    # if np.random.rand() < 0.0002:
                    #     file_json.append(cfg_dict)
                    file_json.append(cfg_dict)

        print(len(file_json))
    with open(os.path.join("../datasets/lpba/test.json"), "w", encoding="utf-8") as f:
        json.dump(file_json, f, indent=4, separators=(",", ":"), ensure_ascii=False)


def LiTS():
    source_path = "../datasets/lits/test.json"
    file_json = []
    with open(source_path, "r", encoding="utf-8") as f:
        source = json.load(f)

    for item in source:
        if np.random.rand() < 0.05:
            file_json.append(item)

    with open("../datasets/lits/val.json", "w", encoding="utf-8") as f:
        json.dump(file_json, f, indent=4, separators=(",", ":"), ensure_ascii=False)


def NLST():
    """
        "fixed": 0,
        "moving": 1
    """
    phase = "Tr"
    path = "../../datasets/NLST/images" + phase
    # idL = []
    # for file in os.listdir(path):
    #     id = file.split("_")[1]
    #     if id not in idL:
    #         idL.append(id)
    # print(len(idL))
    idL = ["{:04}".format(i) for i in range(101, 111)]

    file_json = []
    for idx in idL:
        cfg_dict = {}
        cfg_dict["images_fixed"] = os.path.join("images" + phase, "NLST_" + idx + "_0000.nii.gz").replace("\\", "/")
        cfg_dict["images_moving"] = os.path.join("images" + phase, "NLST_" + idx + "_0001.nii.gz").replace("\\", "/")
        cfg_dict["labels_fixed"] = os.path.join("masks" + phase, "NLST_" + idx + "_0000.nii.gz").replace("\\", "/")
        cfg_dict["labels_moving"] = os.path.join("masks" + phase, "NLST_" + idx + "_0001.nii.gz").replace("\\", "/")
        cfg_dict["points_fixed"] = os.path.join("keypoints" + phase, "NLST_" + idx + "_0000.csv").replace("\\", "/")
        cfg_dict["points_moving"] = os.path.join("keypoints" + phase, "NLST_" + idx + "_0001.csv").replace("\\", "/")
        file_json.append(cfg_dict)
    print(len(file_json))
    with open(os.path.join("../../datasets/NLST", "val.json"), "w", encoding="utf-8") as f:
        json.dump(file_json, f, indent=4, separators=(",", ":"), ensure_ascii=False)


def kpt2mask_csv():
    dataPath = "../datasets/NLST/keypointsTs/NLST_0270_0000.csv"
    df = pd.read_csv(dataPath, header=None)
    points = df.values.astype(np.float32)

    radius = 2  # as RegGraphNet, patch_width = 5 (step = 2)

    # 初始化图像
    image_size = [224, 192, 224]
    image = np.zeros(image_size, dtype=np.uint8)  # 使用uint8类型，适用于二值图像

    # 设置点及其邻域为白色
    for point in points:
        # print(point)
        x, y, z = map(round, point)  # 确保点坐标是整数
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    # 检查邻域点是否在图像范围内
                    if 0 <= x + dx < image_size[0] and 0 <= y + dy < image_size[1] and 0 <= z + dz < image_size[2]:
                        image[z + dz, y + dy, x + dx] = 1

    # image = np.flip(image, axis=1)
    sitk_image = sitk.GetImageFromArray(image)
    # 保存为.nii.gz格式
    nii_file = os.path.join("../p2i.nii.gz")  # 您想保存的文件名
    sitk.WriteImage(sitk_image, nii_file)

    dataPath = "../datasets/NLST/imagesTs/NLST_0270_0000.nii.gz"
    data = sitk.ReadImage(dataPath)
    data = sitk.GetArrayFromImage(data).astype(np.float32)
    data -= data.min()
    data /= data.max()
    sitk_image = sitk.GetImageFromArray(data)
    # 保存为.nii.gz格式
    nii_file = os.path.join("../i2i.nii.gz")  # 您想保存的文件名
    sitk.WriteImage(sitk_image, nii_file)


def kpt2mask_arr(points, trg_path):
    points = points.squeeze().cpu().numpy()
    radius = 2  # as RegGraphNet, patch_width = 5 (step = 2)

    # 初始化图像
    image_size = [224, 192, 224]
    image = np.zeros(image_size, dtype=np.uint8)  # 使用uint8类型，适用于二值图像

    # 设置点及其邻域为白色
    for point in points:
        # print(point)
        x, y, z = map(round, point)  # 确保点坐标是整数
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    # 检查邻域点是否在图像范围内
                    if 0 <= x + dx < image_size[0] and 0 <= y + dy < image_size[1] and 0 <= z + dz < image_size[2]:
                        image[x + dx, y + dy, z + dz] = 1

    # image = np.flip(image, axis=1)
    sitk_image = sitk.GetImageFromArray(image)
    # 保存为.nii.gz格式
    nii_file = os.path.join(trg_path)  # 您想保存的文件名
    sitk.WriteImage(sitk_image, nii_file)
    return sitk_image


def create_mask_from_keypoints(image_shape, keypoints, radius=2):
    """
    根据关键点及其邻域生成mask。
    :param image_shape: 图像的形状
    :param keypoints: 关键点的坐标列表
    :param radius: 邻域的半径
    :return: mask图像
    """
    mask = np.zeros(image_shape, dtype=np.float32)

    for point in keypoints:
        z, y, x = point
        z_min, z_max = int(max(0, z - radius)), int(min(image_shape[0], z + radius + 1))
        y_min, y_max = int(max(0, y - radius)), int(min(image_shape[1], y + radius + 1))
        x_min, x_max = int(max(0, x - radius)), int(min(image_shape[2], x + radius + 1))

        mask[z_min:z_max, y_min:y_max, x_min:x_max] = 1

    return mask


def find_thresh():
    phase = "Ts"
    image_path = "../datasets/NLST/images" + phase
    mask_path = "../datasets/NLST/masks" + phase
    kpt_path = "../datasets/NLST/keypoints" + phase

    all_values = []

    for f in os.listdir(image_path):
        print(f)
        image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(image_path, f))).astype(np.float32)
        ### 用 mask 文件
        # mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_path, f))).astype(np.float32)

        ### 用 kpt 生成 mask
        # 读取关键点
        csv_file = f.replace(".nii.gz", ".csv")
        keypoints_df = pd.read_csv(os.path.join(kpt_path, csv_file))

        # 对点坐标进行flip
        keypoints = keypoints_df.values[:, :3]
        keypoints = keypoints[:, ::-1]  # 假设需要对坐标进行翻转

        # 生成mask
        mask = create_mask_from_keypoints(image.shape, keypoints)

        # 将非零值添加到列表中
        non_zero_values = image[mask != 0].flatten()
        all_values.extend(non_zero_values)

        print(non_zero_values.mean(), non_zero_values.std())

    # 将所有灰度值转换为numpy数组
    all_values = np.array(all_values)

    # 计算均值和标准差
    mean_val = np.mean(all_values)
    std_val = np.std(all_values)

    # 打印均值和标准差
    print(f"Mean: {mean_val:.2f}")
    print(f"Standard Deviation: {std_val:.2f}")

    # 定义区间宽度
    bin_width = 10
    bins = np.arange(np.floor(all_values.min()), np.ceil(all_values.max()) + bin_width, bin_width)

    # 计算直方图
    hist, bin_edges = np.histogram(all_values, bins=bins)

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(all_values, bins=bins, edgecolor='black')
    plt.xlabel('Gray Value')
    plt.ylabel('Frequency')
    plt.title(f'Gray Value Distribution\nMean: {mean_val:.2f}, Std Dev: {std_val:.2f}')

    # 在图上标注均值和标准差
    plt.axvline(mean_val, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')
    plt.axvline(mean_val + std_val, color='g', linestyle='dashed', linewidth=1,
                label=f'Mean + 1 Std: {mean_val + std_val:.2f}')
    plt.axvline(mean_val - std_val, color='g', linestyle='dashed', linewidth=1,
                label=f'Mean - 1 Std: {mean_val - std_val:.2f}')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    LiTS()

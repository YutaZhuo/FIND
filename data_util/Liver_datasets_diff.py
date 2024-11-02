from torch.utils.data import Dataset
import data_util.util_3D as Util
import os
import numpy as np
import torch
import scipy.io as sio
import json
import SimpleITK as sitk


class LiverDataset(Dataset):
    def __init__(self, dataroot, fineSize, split='train'):
        self.split = split
        self.imageNum = []
        self.dataroot = dataroot

        datapath = os.path.join(dataroot, split + '.json')
        with open(datapath, 'r') as f:
            self.imageNum = json.load(f)
        for it in self.imageNum:
            it['image_fixed'] = os.path.join(dataroot, it['image_fixed'])
            it['image_moving'] = os.path.join(dataroot, it['image_moving'])
            it['label_fixed'] = os.path.join(dataroot, it['label_fixed'])
            it['label_moving'] = os.path.join(dataroot, it['label_moving'])

        self.data_len = len(self.imageNum)
        self.fineSize = fineSize

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        dataPath = self.imageNum[index]

        dataA = dataPath['image_fixed']
        dataA = sitk.ReadImage(dataA)
        dataA = sitk.GetArrayFromImage(dataA).astype(np.float32)

        dataB = dataPath['image_moving']
        dataB = sitk.ReadImage(dataB)
        dataB = sitk.GetArrayFromImage(dataB).astype(np.float32)

        label_dataA = dataPath['label_fixed']
        label_dataA = sitk.ReadImage(label_dataA)
        label_dataA = sitk.GetArrayFromImage(label_dataA)
        label_dataB = dataPath['label_moving']
        label_dataB = sitk.ReadImage(label_dataB)
        label_dataB = sitk.GetArrayFromImage(label_dataB)

        label_dataA = (label_dataA > 128).astype(np.float32)
        label_dataB = (label_dataB > 128).astype(np.float32)

        # data normalize, Step1: value range[0, 1]
        dataA -= dataA.min()
        dataA /= dataA.max()
        # dataA /= 255.0

        dataB -= dataB.min()
        dataB /= dataB.max()
        # dataB /= 255.0

        # data normalize, Step2: value range[-1, 1]
        [fixed, moving, fixedM, movingM] = Util.transform_augment([dataA, dataB, label_dataA, label_dataB],
                                                                  split=self.split,
                                                                  min_max=(0, 1))

        return {'M': moving, 'F': fixed, 'MS': movingM, 'FS': fixedM, 'Index': index}

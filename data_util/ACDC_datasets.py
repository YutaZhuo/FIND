from torch.utils.data import Dataset
import data_util.util_3D as Util
import os
import numpy as np
import torch
import json
import SimpleITK as sitk


class ACDCDataset(Dataset):
    def __init__(self, dataroot, fineSize, split='train'):
        self.split = split
        self.imageNum = []
        self.dataroot = dataroot

        datapath = os.path.join(dataroot, split + '.json')
        with open(datapath, 'r') as f:
            self.imageNum = json.load(f)
            print(self.imageNum)
        for it in self.imageNum:
            it['image_ED'] = os.path.join(dataroot, it['image_ED'])
            it['image_ES'] = os.path.join(dataroot, it['image_ES'])
            it['label_ED'] = os.path.join(dataroot, it['label_ED'])
            it['label_ES'] = os.path.join(dataroot, it['label_ES'])

        self.data_len = len(self.imageNum)
        self.fineSize = fineSize

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # ED-fixed, ES-moving
        dataPath = self.imageNum[index]

        dataA = dataPath['image_ED']
        dataA = sitk.ReadImage(dataA)
        dataA = sitk.GetArrayFromImage(dataA).astype(np.float32)

        dataB = dataPath['image_ES']
        dataB = sitk.ReadImage(dataB)
        dataB = sitk.GetArrayFromImage(dataB).astype(np.float32)

        label_dataA = dataPath['label_ED']
        label_dataA = sitk.ReadImage(label_dataA)
        label_dataA = sitk.GetArrayFromImage(label_dataA)
        label_dataB = dataPath['label_ES']
        label_dataB = sitk.ReadImage(label_dataB)
        label_dataB = sitk.GetArrayFromImage(label_dataB)

        # data normalize, Step1: value range[0, 1]
        dataA -= dataA.min()
        dataA /= dataA.max()

        dataB -= dataB.min()
        dataB /= dataB.max()

        # data normalize, Step2: value range[-1, 1]
        [fixed, moving, fixedM, movingM] = Util.transform_augment([dataA, dataB, label_dataA, label_dataB],
                                                                  split=self.split,
                                                                  min_max=(0, 1))

        return {'M': moving, 'F': fixed, 'MS': movingM, 'FS': fixedM, 'Index': index,
                'name': dataPath['image_ES'].split('/')[3].split("\\")[1] + "--" + dataPath['image_ED'].split('/')[3].split("\\")[1]}

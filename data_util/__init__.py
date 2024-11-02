import os
import torch.utils.data


class Path:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)

    def __call__(self, *names):
        return os.path.join(*((self.path,) + names))


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'test' or phase == "val":
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset_Liver(dataroot, finesize, phase):
    '''create dataset'''
    from data_util.Liver_datasets import LiverDataset as D
    dataset = D(dataroot=dataroot,
                fineSize=finesize,
                split=phase,
                )
    return dataset


def create_dataset_Brain(dataroot, finesize, phase):
    '''create dataset'''
    from data_util.Brain_datasets import BrainDataset as D
    dataset = D(dataroot=dataroot,
                fineSize=finesize,
                split=phase,
                )
    return dataset


def create_dataset_OASIS(dataroot, finesize, phase):
    '''create dataset'''
    from data_util.OASIS_datasets import OASISDataset as D
    dataset = D(dataroot=dataroot,
                fineSize=finesize,
                split=phase,
                )
    return dataset


def create_dataset_ACDC(dataroot, finesize, phase):
    '''create dataset'''
    from data_util.ACDC_datasets import ACDCDataset as D
    dataset = D(dataroot=dataroot,
                fineSize=finesize,
                split=phase,
                )
    return dataset


def create_dataset_NLST(dataroot, finesize, phase):
    '''create dataset'''
    from data_util.NLST_datasets import NLSTDataset as D
    dataset = D(dataroot=dataroot,
                fineSize=finesize,
                split=phase,
                )
    return dataset

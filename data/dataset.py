from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import torch
from torchvision.datasets import CIFAR10, CIFAR100, STL10
from PIL import Image
import numpy as np

class CustomCIFAR10(CIFAR10):
    def __init__(self, withLabel=False, labelSubSet=None, labelTrans=None, **kwds):
        super().__init__(**kwds)
        self.withLabel = withLabel
        self.labelTrans = labelTrans

        if labelSubSet is not None:
            self.data = self.data[labelSubSet]

    def __getitem__(self, idx):
        # if not self.train:
        #     return super().__getitem__(idx)

        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        if not self.withLabel:
            return torch.stack(imgs)
        else:
            imgLabelTrans = self.labelTrans(img)
            label = self.targets[idx]
            return torch.stack(imgs), imgLabelTrans, label

class CustomCIFAR100(CIFAR100):
    def __init__(self, withLabel=False, labelTrans=None, **kwds):
        super().__init__(**kwds)
        self.withLabel = withLabel
        self.labelTrans = labelTrans

    def __getitem__(self, idx):
        # if not self.train:
        #     return super().__getitem__(idx)

        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        if not self.withLabel:
            return torch.stack(imgs)
        else:
            imgLabelTrans = self.labelTrans(img)
            label = self.targets[idx]
            return torch.stack(imgs), imgLabelTrans, label

class CustomSTL10(STL10):
    def __init__(self, withLabel=False, labelTrans=None, **kwds):
        super().__init__(**kwds)
        self.withLabel = withLabel
        self.labelTrans = labelTrans

    def __getitem__(self, idx):
        # if not self.train:
        #     return super().__getitem__(idx)

        img = self.data[idx]
        img = Image.fromarray(np.transpose(img, (1, 2, 0))).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        if not self.withLabel:
            return torch.stack(imgs)
        else:
            assert False

class CIFAR10IndexPseudoLabelEnsemble(Dataset):
    def __init__(self, root='', transform=None, download=False, train=True,
                 pseudoLabel=None):
        self.cifar10 = datasets.CIFAR10(root=root,
                                        download=download,
                                        train=train,
                                        transform=transform)

        self.pseudo_label = pseudoLabel

    def __getitem__(self, index):
        data, target = self.cifar10[index]
        label = self.pseudo_label[index]

        return data, target, label, index

    def __len__(self):
        return len(self.cifar10)

class CIFAR100IndexPseudoLabelEnsemble(Dataset):
    def __init__(self, root='', transform=None, download=False, train=True,
                 pseudoLabel=None):
        self.cifar100 = datasets.CIFAR100(root=root,
                                        download=download,
                                        train=train,
                                        transform=transform)

        self.pseudo_label = pseudoLabel

    def __getitem__(self, index):
        data, target = self.cifar100[index]
        label = self.pseudo_label[index]

        return data, target, label, index

    def __len__(self):
        return len(self.cifar100)
    
class STL10IndexPseudoLabelEnsemble(Dataset):
    def __init__(self, root='', transform=None, download=False, split='unlabeled',
                 pseudoLabel=None):
        self.stl10 = datasets.STL10(root=root,
                                    download=download,
                                    split=split,
                                    transform=transform)

        self.pseudo_label = pseudoLabel

    def __getitem__(self, index):
        data = self.stl10[index][0]
        label = self.pseudo_label[index]

        return data, label, label, index

    def __len__(self):
        return len(self.stl10)



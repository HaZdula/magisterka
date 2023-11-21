import os
import os.path

import numpy as np
import torchvision

from PIL import Image

import torch.utils.data as data

from torchvision.datasets.utils import download_url, check_integrity
import os
import os.path

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity


def load_new_test_data(root, version='default'):
    data_path = root
    filename = 'cifar10.1'
    if version == 'default':
        pass
    elif version == 'v0':
        filename += '-v0'
    else:
        raise ValueError('Unknown dataset version "{}".'.format(version))
    label_filename = filename + '-labels.npy'
    imagedata_filename = filename + '-data.npy'
    label_filepath = os.path.join(data_path, label_filename)
    imagedata_filepath = os.path.join(data_path, imagedata_filename)
    labels = np.load(label_filepath).astype(np.int64)
    imagedata = np.load(imagedata_filepath)
    assert len(labels.shape) == 1
    assert len(imagedata.shape) == 4
    assert labels.shape[0] == imagedata.shape[0]
    assert imagedata.shape[1] == 32
    assert imagedata.shape[2] == 32
    assert imagedata.shape[3] == 3
    if version == 'default':
        assert labels.shape[0] == 2000
    elif version == 'v0':
        assert labels.shape[0] == 2021
    return imagedata, labels


class CIFAR10_1(data.Dataset):
    images_url = 'https://github.com/modestyachts/CIFAR-10.1/blob/master/datasets/cifar10.1_v6_data.npy?raw=true'
    images_md5 = '4fcae82cb1326aec9ed1dc1fc62345b8'
    images_filename = 'cifar10.1-data.npy'

    labels_url = 'https://github.com/modestyachts/CIFAR-10.1/blob/master/datasets/cifar10.1_v6_labels.npy?raw=true'
    labels_md5 = '09a97fb7c430502fcbd69d95093a3f85'
    labels_filename = 'cifar10.1-labels.npy'

    classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
        'ship', 'truck'
    ]

    @property
    def targets(self):
        return self.labels

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        images, labels = load_new_test_data(root)

        self.data = images
        self.labels = labels

        self.class_to_idx = {
            _class: i
            for i, _class in enumerate(self.classes)
        }

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        data_path = os.path.join(self.root, self.images_filename)
        labels_path = os.path.join(self.root, self.labels_filename)

        return (check_integrity(data_path, self.images_md5) and
            check_integrity(labels_path, self.labels_md5))

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.images_url, root, self.images_filename)
        download_url(self.labels_url, root, self.labels_filename)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp,
            self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp,
            self.target_transform.__repr__().replace('\n',
                                                     '\n' + ' ' * len(tmp)))
        return fmt_str

#x = CIFAR10_1("/home/hania/studia/mag", download=True)




# STL dataset, but remove label not in CIFAR-10, and remap to CIFAR-10 labels.
# Adapted from https://github.com/pytorch/vision/blob/master/torchvision/datasets/stl10.py
# We copy this instead of just modifying .data .labels in case the code gets modified.

from PIL import Image
import os
import os.path
import numpy as np
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, verify_str_arg


class STL10(torchvision.datasets.STL10):

    def make_labels_like_cifar(self):
        stl_to_cifar_indices = np.array([0, 2, 1, 3, 4, 5, 7, -1, 8, 9])
        self.labels = stl_to_cifar_indices[self.labels]
        indices = np.where(self.labels != -1)
        self.labels = self.labels[indices]
        self.data = self.data[indices]
        new_classes = ['']*10
        new_classes[6] = 'NONE'
        for i in range(len(stl_to_cifar_indices)):
            v = stl_to_cifar_indices[i]
            if v != -1:
                new_classes[v] = self.classes[i]
        self.classes = new_classes

    def __init__(
            self,
            root: str,
            split: str = "train",
            folds: Optional[int] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(STL10, self).__init__(root, transform=transform,
                                    target_transform=target_transform, download=download)
        self.split = verify_str_arg(split, "split", self.splits)
        self.folds = self._verify_folds(folds)

        if download:
            self.download()
        elif not self._check_integrity():
            raise RuntimeError(
                'Dataset not found or corrupted. '
                'You can use download=True to download it')

        # Make labels like CIFAR.
        self.make_labels_like_cifar()

"""Return training and evaluation/test datasets from config files."""
import json
import os
import os.path
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as vdsets
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler, TensorDataset


def identity(x):
    return x

def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data

# fast class to load all images
class ImageFolderFast(vdsets.VisionDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.image_paths = os.listdir(root)
        self.transform = transform

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.image_paths[index])
        with open(image_path, "rb") as f:
            img = Image.open(f)
            x = img.convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        return x, #needed to make it consistent: index dataset[0][0] for image

    def __len__(self):
        return len(self.image_paths)

# fast class to load all images
class ImageFolderClassFast(vdsets.VisionDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        with open(os.path.join(root, "dataset.json"), "r") as f:
            self.image_paths = json.load(f)["labels"]
        self.transform = transform

    def __getitem__(self, index):
        pair = self.image_paths[index]
        image_path = os.path.join(self.root, pair[0])
        with open(image_path, "rb") as f:
            img = Image.open(f)
            x = img.convert("RGB")
        if self.transform is not None:
            x = self.transform(x) 
        return x, pair[1]

    def __len__(self):
        return len(self.image_paths)


def get_dataset(config, evaluation=False, distributed=True):
    
    dataroot = config.dataroot
    if config.data.dataset == "CIFAR10":
        
        train_transforms = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip() if config.data.random_flip else identity,
                transforms.ToTensor(),
            ]
        )
        test_transforms = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.ToTensor(),
            ]
        )

        train_set = vdsets.CIFAR10(dataroot, train=True, transform=train_transforms)
        test_set = vdsets.CIFAR10(dataroot, train=False, transform=test_transforms)
        workers = 2
    elif config.data.dataset == "ImageNet32":
        data_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        train_set = ImageFolderFast(os.path.join(dataroot, "ds_imagenet", "train_32x32"), transform=data_transforms)
        test_set = ImageFolderFast(os.path.join(dataroot, "ds_imagenet", "valid_32x32"), transform=data_transforms)
        workers = 4
    elif config.data.dataset == "ImageNet64C":
        data_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        train_set = ImageFolderClassFast(os.path.join(dataroot, "imagenet-64x64", "train"), transform=data_transforms)
        test_set = ImageFolderClassFast(os.path.join(dataroot, "imagenet-64x64", "valid"), transform=data_transforms)
        workers = 4
    else:
        raise ValueError(f"{config.data.dataset} is not valid")

    if evaluation:
        if distributed:
            sampler = DistributedSampler(test_set, shuffle=False)
        else:
            sampler = None

        test_loader = DataLoader(
            test_set,
            batch_size=config.eval.batch_size,
            sampler=sampler,
            num_workers=workers,
            pin_memory=True,
            shuffle=(sampler is None)
        )

        return test_loader
    else:
        if config.training.batch_size % config.ngpus != 0:
            raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus.")
        if config.eval.batch_size % config.ngpus != 0:
            raise ValueError(f"Eval Batch Size {config.eval.batch_size} is not divisible by {config.ngpus} gpus.")
    
        if distributed:
            train_sampler = DistributedSampler(train_set) 
            test_sampler = DistributedSampler(test_set)
        else:
            train_sampler = None
            test_sampler = None

        train_loader = DataLoader(
            train_set,
            batch_size=config.training.batch_size // config.ngpus,
            sampler=train_sampler,
            num_workers=workers,
            pin_memory=True,
            shuffle=(train_sampler is None),
            persistent_workers=True if workers > 0 else False,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=config.eval.batch_size // config.ngpus,
            sampler=test_sampler,
            num_workers=workers,
            pin_memory=True,
            shuffle=(test_sampler is None),
        )

        train_loader, test_loader = cycle_loader(train_loader, train_sampler), cycle_loader(test_loader, test_sampler)
        return train_loader, test_loader

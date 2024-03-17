import torch
from torch.utils.data import ConcatDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import os
from pathlib import Path
import shutil
import random

classes = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def prepare_data(n_splits=10):
    random.seed(42)
    valid_path = Path("proj1/data/valid")
    train_path = Path("proj1/data/train")
    for cls in classes:
        cls_valid_path = valid_path / cls
        cls_train_path = train_path / cls
        for file in cls_valid_path.iterdir():
            shutil.copy(file, cls_train_path)
    for cls in classes:
        cls_train_path = train_path / cls
        files = set(cls_train_path.iterdir())
        files_per_split = len(files) // n_splits
        for split in range(n_splits - 1):
            dest = train_path / str(split) / cls
            os.makedirs(dest, exist_ok=True)
            split_files = random.sample(list(files), k=files_per_split)
            files -= set(split_files)
            for file in split_files:
                shutil.move(file, dest)

        dest = train_path / str(n_splits - 1) / cls
        os.makedirs(dest, exist_ok=True)
        for file in files:
            shutil.move(file, dest)


def get_cv_data_loaders(
    train_transforms=None,
    valid_transforms=None,
    normalize=False,
    batch_size=512,
    n_splits=10,
    num_workers=11,
):
    cinic_directory = "proj1/data"
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    dataloader_setting = {"batch_size": batch_size, "num_workers": num_workers}

    train_transform_list = [transforms.ToTensor()]
    valid_transform_list = [transforms.ToTensor()]

    if train_transforms:
        train_transform_list += train_transforms

    if valid_transforms:
        valid_transform_list += valid_transforms

    if normalize:
        train_transform_list.append(transforms.Normalize(cinic_mean, cinic_std))
        valid_transform_list.append(transforms.Normalize(cinic_mean, cinic_std))

    train_transform = transforms.Compose(train_transform_list)
    valid_transform = transforms.Compose(valid_transform_list)

    test_dataset = torchvision.datasets.ImageFolder(
        cinic_directory + f"/test/", transform=valid_transform
    )

    test_dataloader = DataLoader(test_dataset, shuffle=False, **dataloader_setting)

    for valid_split_idx in range(n_splits):
        train_dataset_list = [
            torchvision.datasets.ImageFolder(
                cinic_directory + f"/train/{split_idx}", transform=train_transform
            )
            for split_idx in range(n_splits)
            if split_idx != valid_split_idx
        ]
        train_dataset = ConcatDataset(train_dataset_list)

        train_dataloader = DataLoader(train_dataset, shuffle=True, **dataloader_setting)

        valid_dataset = torchvision.datasets.ImageFolder(
            cinic_directory + f"/train/{valid_split_idx}", transform=valid_transform
        )

        valid_dataloader = DataLoader(
            valid_dataset, shuffle=False, **dataloader_setting
        )

        yield train_dataloader, valid_dataloader, test_dataloader

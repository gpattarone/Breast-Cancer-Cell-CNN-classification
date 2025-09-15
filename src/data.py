# -*- coding: utf-8 -*-
"""
data.py - Data loading utilities for Breast Cancer Cell Classification
---------------------------------------------------------------------
Defines PyTorch Dataset and DataLoader pipelines using torchvision.
"""

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(img_size=64):
    """Return train/val/test transforms."""
    train_tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270))
        ]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    eval_tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return {"train": train_tf, "val": eval_tf, "test": eval_tf}


def get_dataloaders(data_root="data", img_size=64, batch_size=16, num_workers=2):
    """
    Create dataloaders for train/val/test splits.

    Args:
        data_root (str): path to dataset root (expects train/, val/, test/ subfolders).
        img_size (int): image size for resizing.
        batch_size (int): batch size for training.
        num_workers (int): workers for DataLoader.

    Returns:
        dataloaders (dict): dict with train/val/test dataloaders.
        dataset_sizes (dict): dict with dataset sizes.
        class_names (list): list of class labels.
    """
    root = Path(data_root)
    tfs = get_transforms(img_size)

    image_datasets = {
        split: datasets.ImageFolder(root / split, tfs[split])
        for split in ["train", "val", "test"]
    }

    dataloaders = {
        split: DataLoader(image_datasets[split],
                          batch_size=batch_size,
                          shuffle=(split == "train"),
                          num_workers=num_workers,
                          drop_last=True)
        for split in ["train", "val", "test"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "test"]}
    class_names = image_datasets["train"].classes

    return dataloaders, dataset_sizes, class_names


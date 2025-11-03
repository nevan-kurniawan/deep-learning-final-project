"""
Data loading utilities for the Describable Textures Dataset (DTD).

This module defines the `create_dataloaders` function, which is responsible
for instantiating and configuring the train, validation, and test
DataLoaders for the DTD dataset.
"""

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from pathlib import Path

def create_dataloaders(
    data_dir: Path,
    batch_size: int,
    image_size: int = 224,
    num_workers: int = 4,
    splits: list[str] = None
) -> tuple[dict[str, DataLoader], list[str]]:
    """
    Creates DataLoaders for the DTD dataset using torchvision.

    Args:
        data_dir: Directory where the dataset would be contained.
        batch_size: The chosen batch size.
        image_size: The size of the image to use. Defaults to 224.
        num_workers: The number of CPU workers to use. Defaults to 4.
        splits: The splits to use for train/val/test split. Defaults to None. If none, creates split for all three.

    Returns:
        A tuple containing:
            - A dictionary mapping split names (e.g., 'train') to their
              respective DataLoader instances.
            - A list of the dataset's class names.
    """
    if splits is None:
        splits = ['train', 'val', 'test']

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_loaders = {}
    class_names = []

    data_dir.mkdir(parents=True, exist_ok=True)

    persistent_workers = num_workers > 0

    if 'train' in splits:
        train_dataset = datasets.DTD(
            root=str(data_dir),
            split='train',
            download=True,
            transform=train_transform
        )
        data_loaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers
        )
        class_names = train_dataset.classes # Get class names from the dataset object

    if 'val' in splits:
        val_dataset = datasets.DTD(
            root=str(data_dir),
            split='val',
            download=True,
            transform=val_test_transform
        )
        data_loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False, # No shuffle for validation
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers
        )
        if not class_names: # Get class names if train set wasn't loaded
             class_names = val_dataset.classes

    if 'test' in splits:
        test_dataset = datasets.DTD(
            root=str(data_dir),
            split='test',
            download=True,
            transform=val_test_transform
        )
        data_loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False, # No shuffle for test
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers
        )
        if not class_names: # Get class names if train/val weren't loaded
             class_names = test_dataset.classes

    if not class_names:
        raise RuntimeError("Could not obtain class names from the DTD dataset.")

    return data_loaders, class_names
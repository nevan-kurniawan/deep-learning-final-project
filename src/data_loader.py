import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class PetDataset(Dataset):
    """Custom Dataset for the Oxford-IIIT Pet Dataset."""

    def __init__(self, manifest_path: Path, transform=None):
        """
        Initializes the Dataset.

        Parameters
        ----------
        manifest_path : pathlib.Path
            Path to the .csv manifest file (train.csv, test.csv, val.csv).
        transform : torchvision.transforms.Compose
            A composition of transforms to apply to the images.
        """
        self.manifest = pd.read_csv(manifest_path)
        self.transform = transform

        #list of classes taken from the manifest file label
        self.classes = sorted(self.manifest['label'].unique())
        #convert class to idx. Example: Abyssinian to 0, american_bulldog to 1, etc.
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        """
        Fetches a single sample from the dataset at the given index.

        Returns
        -------
        tuple[torch.Tensor, int, str]
            A tuple containing the transformed image, its integer label, and its filepath.
        """
        img_path_str = PROJECT_ROOT / self.manifest.iloc[idx]['filepath']

        label_str = self.manifest.iloc[idx]['label']

        image = Image.open(img_path_str).convert("RGB")
        label = self.class_to_idx[label_str]

        if self.transform:
            image = self.transform(image)

        # Return the filepath along with the image and label
        return image, label, str(img_path_str)


def create_dataloaders(data_dir: Path, batch_size: int, image_size: int = 224, num_workers: int = 4, splits: list[str] = None) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Creates training, validation, and test DataLoaders.

    Parameters
    ----------
    splits
    data_dir : pathlib.Path
        The path to the 'processed' data directory containing the .csv manifests.
    batch_size : int
        The number of samples per batch.
    image_size : int, optional
        The target size for the images, by default 224.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader, list[str]]
        A tuple containing the train, validation, and test DataLoaders, and the list of class names.
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

    if 'train' in splits:
        train_dataset = PetDataset(manifest_path=data_dir / "train.csv", transform=train_transform)
        data_loaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
        class_names = train_dataset.classes  # Get class names from train set

    if 'val' in splits:
        val_dataset = PetDataset(manifest_path=data_dir / "val.csv", transform=val_test_transform)
        data_loaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True)

    if 'test' in splits:
        test_dataset = PetDataset(manifest_path=data_dir / "test.csv", transform=val_test_transform)
        data_loaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    return tuple(data_loaders.get(s) for s in splits) + (class_names,)
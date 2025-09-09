import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path


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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Fetches a single sample from the dataset at the given index.

        Parameters
        ----------
        idx : int
            The index of the sample to fetch.

        Returns
        -------
        tuple[torch.Tensor, int]
            A tuple containing the transformed image tensor and its integer label.
        """
        img_path_str = self.manifest.iloc[idx]['filepath']
        label_str = self.manifest.iloc[idx]['label']

        image = Image.open(img_path_str).convert("RGB")

        label = self.class_to_idx[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataloaders(
        data_dir: Path,
        batch_size: int,
        image_size: int = 224
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Creates training, validation, and test DataLoaders.

    Parameters
    ----------
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
    # Define the image transformations. These should be informed by your EDA.
    # Normalization values are standard for ImageNet-pretrained models.
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # Add augmentations justified by your EDA here (e.g., RandomRotation, ColorJitter)
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create Dataset instances for each split.
    train_dataset = PetDataset(manifest_path=data_dir / "train.csv", transform=train_transform)
    val_dataset = PetDataset(manifest_path=data_dir / "val.csv", transform=val_test_transform)
    test_dataset = PetDataset(manifest_path=data_dir / "test.csv", transform=val_test_transform)

    # Create DataLoader instances.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(),
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(),
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(),
                             pin_memory=True)

    return train_loader, val_loader, test_loader, train_dataset.classes
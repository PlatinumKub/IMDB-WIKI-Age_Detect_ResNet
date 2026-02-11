from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class CustomImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_root: Path | str,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        df: DataFrame с колонками ['image', 'age'].
        images_root: корневая папка, относительно которой интерпретируется 'image'
                     (например, data/raw).
        """
        self.df = df.reset_index(drop=True)
        self.images_root = Path(images_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        rel_path = row["image"]
        age = row["age"]

        image_path = self.images_root / rel_path
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        age = torch.tensor(age, dtype=torch.float32)
        return image, age


def get_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    images_root: Path | str,
    batch_size: int = 256,
    num_workers: int = 8,
):
    train_dataset = CustomImageDataset(
        train_df, images_root=images_root, transform=get_train_transforms()
    )
    val_dataset = CustomImageDataset(
        val_df, images_root=images_root, transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader

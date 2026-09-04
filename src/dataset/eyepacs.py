import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from pathlib import Path
from PIL import Image
import pandas as pd
import torch


class EyePACSDataset(Dataset):
    """EyePACS 5-class DR grading (Grade 0–4)."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class EyePACSDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.image_size = cfg.dataset.image_size
        self.batch_size = cfg.dataset.batch_size
        self.num_workers = cfg.dataset.num_workers
        self.data_root = Path(cfg.dataset.data_root)
        self.seed = cfg.seed

    def _resolve_paths(self, df, img_dir):
        """Match CSV rows to actual image files."""
        paths, labels = [], []
        for _, row in df.iterrows():
            name = row["image"]
            for ext in [".jpeg", ".jpg", ".png", ""]:
                p = img_dir / f"{name}{ext}"
                if p.exists():
                    paths.append(p)
                    labels.append(int(row["level"]))
                    break
        return paths, torch.tensor(labels, dtype=torch.long)

    def setup(self, stage=None):
        csv_path = self.data_root / "trainLabels.csv"
        df = pd.read_csv(csv_path)

        # Find image directory — flat structure from zip or train/ subfolder
        img_dir = self.data_root / "train"
        if not img_dir.exists():
            # find the dir that actually contains the .jpeg files
            img_dir = next(
                p.parent for p in self.data_root.rglob("*.jpeg") if p.is_file()
            )

        paths, labels = self._resolve_paths(df, img_dir)

        # 8:1:1 split
        n = len(paths)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        gen = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(n, generator=gen).tolist()

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        mean = std = [0.5, 0.5, 0.5]
        train_transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.RandomResizedCrop(self.image_size, scale=(0.75, 1.0), ratio=(0.95, 1.05)),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        eval_transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

        self.train_ds = EyePACSDataset(
            [paths[i] for i in train_idx], labels[train_idx], train_transform,
        )
        self.val_ds = EyePACSDataset(
            [paths[i] for i in val_idx], labels[val_idx], eval_transform,
        )
        self.test_ds = EyePACSDataset(
            [paths[i] for i in test_idx], labels[test_idx], eval_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers, pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True,
        )

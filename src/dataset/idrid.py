import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as T
from pathlib import Path
from PIL import Image
import pandas as pd
import torch


class IDRiDDataset(Dataset):
    """IDRiD 5-class DR grading (Grade 0–4)."""

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


class IDRiDDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.image_size = cfg.dataset.image_size
        self.batch_size = cfg.dataset.batch_size
        self.num_workers = cfg.dataset.num_workers
        self.data_root = Path(cfg.dataset.data_root)
        self.seed = cfg.seed

    def _find_grading_root(self):
        """Locate the Disease Grading subtree (handles URL-encoded dir names)."""
        for d in self.data_root.iterdir():
            if "disease" in d.name.lower().replace("%20", " ") or "grading" in d.name.lower().replace("%20", " "):
                # one more nesting level (e.g. "B. Disease Grading/B. Disease Grading")
                for sub in d.iterdir():
                    if sub.is_dir():
                        return sub
                return d
        return self.data_root

    def _load_split(self, grading, split="train"):
        """Load image paths and labels for a split ('train' or 'test')."""
        keyword = "Training" if split == "train" else "Testing"

        csv_path = next(grading.rglob(f"*{keyword}*Labels*.csv"))
        df = pd.read_csv(csv_path)

        cols = df.columns.tolist()
        img_col = [c for c in cols if "image" in c.lower() or "img" in c.lower()][0]
        grade_col = [c for c in cols if "retinopathy" in c.lower() or "grade" in c.lower() or "dr" in c.lower()][0]

        df = df[df[grade_col].isin([0, 1, 2, 3, 4])].reset_index(drop=True)

        # Find image directory for this split
        img_dir = next(
            p for p in grading.rglob(f"*{keyword}*")
            if p.is_dir() and "original" in str(p).lower()
        )

        paths, labels = [], []
        for _, row in df.iterrows():
            name = row[img_col]
            for ext in [".jpg", ".jpeg", ".png", ".tif", ""]:
                p = img_dir / f"{name}{ext}"
                if p.exists():
                    paths.append(p)
                    labels.append(int(row[grade_col]))
                    break

        return paths, torch.tensor(labels, dtype=torch.long)

    def setup(self, stage=None):
        grading = self._find_grading_root()

        # Official train split → split into train/val (9:1)
        paths, labels = self._load_split(grading, "train")
        n = len(paths)
        n_val = max(1, int(n * 0.1))
        n_train = n - n_val

        gen = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(n, generator=gen).tolist()
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        # Official test split
        test_paths, test_labels = self._load_split(grading, "test")

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

        # Inverse-frequency class weights from training split
        train_labels = labels[train_idx]
        counts = torch.bincount(train_labels, minlength=5).float()
        self.class_weights = (1.0 / counts) * counts.sum() / len(counts)

        self.train_ds = IDRiDDataset(
            [paths[i] for i in train_idx], labels[train_idx], train_transform,
        )
        self.val_ds = IDRiDDataset(
            [paths[i] for i in val_idx], labels[val_idx], eval_transform,
        )
        self.test_ds = IDRiDDataset(test_paths, test_labels, eval_transform)

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

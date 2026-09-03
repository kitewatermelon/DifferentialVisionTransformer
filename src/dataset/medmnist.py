import medmnist
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as T


def _get_medmnist_class(name: str):
    """Resolve MedMNIST dataset class from dataset name (e.g. 'organamnist')."""
    info = medmnist.INFO.get(name)
    if info is None:
        raise ValueError(f"Unknown MedMNIST dataset: '{name}'. Available: {list(medmnist.INFO)}")
    return getattr(medmnist, info["python_class"])


class MedMNISTDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.image_size = cfg.dataset.image_size
        self.batch_size = cfg.dataset.batch_size
        self.num_workers = cfg.dataset.num_workers
        self.data_root = getattr(cfg.dataset, "data_root", None)
        self._ds_class = _get_medmnist_class(cfg.dataset.name)

    def setup(self, stage=None):
        mean = std = [0.5, 0.5, 0.5]
        train_transform = T.Compose([
            T.RandomResizedCrop(self.image_size, scale=(0.75, 1.0), ratio=(0.95, 1.05)),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        eval_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        kwargs = dict(download=True, size=self.image_size, as_rgb=True)
        if self.data_root is not None:
            kwargs["root"] = self.data_root
        self.train_ds = self._ds_class(split="train", transform=train_transform, **kwargs)
        self.val_ds = self._ds_class(split="val", transform=eval_transform, **kwargs)
        self.test_ds = self._ds_class(split="test", transform=eval_transform, **kwargs)

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


# backward compat
PathMNISTDataModule = MedMNISTDataModule

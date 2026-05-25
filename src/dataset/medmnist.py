import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as T
from medmnist import PathMNIST


class PathMNISTDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.image_size = cfg.dataset.image_size
        self.batch_size = cfg.dataset.batch_size
        self.num_workers = cfg.dataset.num_workers
        self.data_root = getattr(cfg.dataset, "data_root", None)

    def setup(self, stage=None):
        mean = std = [0.5, 0.5, 0.5]
        train_transform = T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.Normalize(mean, std),
        ])
        eval_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        kwargs = dict(download=True, size=self.image_size, as_rgb=True)
        if self.data_root is not None:
            kwargs["root"] = self.data_root
        self.train_ds = PathMNIST(split="train", transform=train_transform, **kwargs)
        self.val_ds = PathMNIST(split="val", transform=eval_transform, **kwargs)
        self.test_ds = PathMNIST(split="test", transform=eval_transform, **kwargs)

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

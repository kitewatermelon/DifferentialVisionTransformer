import pytest
from omegaconf import OmegaConf


@pytest.fixture
def cfg():
    return OmegaConf.create({
        "dataset": {
            "name": "pathmnist",
            "image_size": 224,
            "batch_size": 4,
            "num_workers": 0,
        }
    })


def test_datamodule_instantiates(cfg):
    from dataset.medmnist import MedMNISTDataModule
    dm = MedMNISTDataModule(cfg)
    assert dm.batch_size == 4
    assert dm.image_size == 224
    assert dm.num_workers == 0


@pytest.mark.slow
def test_datamodule_dataloaders_shape(cfg):
    import torch
    from dataset.medmnist import MedMNISTDataModule
    dm = MedMNISTDataModule(cfg)
    dm.setup()
    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == (4, 3, 224, 224)
    assert y.shape[0] == 4
    assert x.dtype == torch.float32

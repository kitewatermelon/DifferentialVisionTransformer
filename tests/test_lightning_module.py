import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture
def cfg():
    return OmegaConf.create({
        "model": {"name": "vit", "backbone": "vit_tiny_patch16_224", "num_classes": 9},
        "trainer": {"lr": 1e-4, "max_epochs": 100},
    })


@pytest.fixture
def module(cfg):
    from model.builder import build_model
    from lightning_module import VitLitModule
    model = build_model(cfg.model)
    return VitLitModule(model, cfg)


def test_forward_output_shape(module):
    x = torch.randn(2, 3, 224, 224)
    out = module(x)
    assert out.shape == (2, 9)


def test_training_step_returns_scalar_loss(module):
    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 9, (2, 1))
    loss = module.training_step((x, y), batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_configure_optimizers_has_scheduler(module):
    result = module.configure_optimizers()
    assert "optimizer" in result
    assert "lr_scheduler" in result
    assert isinstance(result["optimizer"], torch.optim.AdamW)

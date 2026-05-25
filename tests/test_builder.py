import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture
def base_cfg():
    return OmegaConf.create({
        "backbone": "vit_tiny_patch16_224",
        "num_classes": 9,
    })


@pytest.mark.parametrize("name", ["vit", "diff_vit", "a6d6", "odd_diff_vit", "even_diff_vit"])
def test_build_model_all_variants(base_cfg, name):
    from model.builder import build_model
    cfg = OmegaConf.merge(base_cfg, {"name": name})
    model = build_model(cfg)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 9)


def test_build_model_invalid_name_raises(base_cfg):
    from model.builder import build_model
    cfg = OmegaConf.merge(base_cfg, {"name": "nonexistent"})
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(cfg)

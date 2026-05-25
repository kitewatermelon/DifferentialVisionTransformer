import torch.nn as nn
from model.vit import get_vit
from model.diff_vit import get_diff_vit, get_a6d6, get_odd_diff_vit, get_even_diff_vit

_BUILDERS = {
    "vit": get_vit,
    "diff_vit": get_diff_vit,
    "a6d6": get_a6d6,
    "odd_diff_vit": get_odd_diff_vit,
    "even_diff_vit": get_even_diff_vit,
}


def build_model(cfg) -> nn.Module:
    if cfg.name not in _BUILDERS:
        raise ValueError(f"Unknown model: '{cfg.name}'. Choose from {list(_BUILDERS)}")
    model = _BUILDERS[cfg.name](cfg.backbone)
    model.reset_classifier(cfg.num_classes)
    return model

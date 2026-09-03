import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from model.builder import build_model
from dataset.medmnist import MedMNISTDataModule
from lightning_module import VitLitModule
from utils.vis_attn import log_attn_maps


@hydra.main(config_path="configs", config_name="config", version_base=None)
def test(cfg: DictConfig) -> None:
    ckpt_dir = (
        Path(get_original_cwd())
        / "outputs"
        / "checkpoints"
        / cfg.dataset.name
        / cfg.model.name
        / f"lr{cfg.trainer.lr}_bs{cfg.dataset.batch_size}_ep{cfg.trainer.max_epochs}"
    )

    ckpt_path = ckpt_dir / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    saved_cfg = OmegaConf.load(ckpt_dir / "config.yaml")
    model = build_model(saved_cfg.model)
    dm = MedMNISTDataModule(saved_cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    module = VitLitModule(model=model, cfg=saved_cfg)
    module.load_state_dict(ckpt["state_dict"])

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        tags=list(cfg.wandb.tags),
        log_model=False,
    )

    trainer = pl.Trainer(
        accelerator=saved_cfg.trainer.accelerator,
        precision=saved_cfg.trainer.precision,
        logger=wandb_logger,
    )

    trainer.test(module, dm)

    dm.setup("test")
    sample_x, _ = next(iter(dm.test_dataloader()))
    sample_x = sample_x.to("cuda" if saved_cfg.trainer.accelerator == "gpu" else "cpu")
    module.model.to(sample_x.device)
    log_attn_maps(module.model, sample_x, wandb_logger)


if __name__ == "__main__":
    test()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from model.builder import build_model
from dataset.medmnist import PathMNISTDataModule
from lightning_module import VitLitModule


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    model = build_model(cfg.model)
    dm = PathMNISTDataModule(cfg)
    module = VitLitModule(model, cfg)

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        tags=list(cfg.wandb.tags),
        log_model=False,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        precision=cfg.trainer.precision,
        logger=wandb_logger,
        log_every_n_steps=10,
    )

    trainer.fit(module, dm)
    trainer.test(module, dm)


if __name__ == "__main__":
    train()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from model.builder import build_model
from dataset.medmnist import MedMNISTDataModule
from lightning_module import VitLitModule


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    model = build_model(cfg.model)
    dm = MedMNISTDataModule(cfg)
    module = VitLitModule(model, cfg)

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        tags=list(cfg.wandb.tags),
        log_model=False,
    )

    ckpt_dir = (
        Path(get_original_cwd())
        / "outputs"
        / "checkpoints"
        / cfg.dataset.name
        / cfg.model.name
        / f"lr{cfg.trainer.lr}_bs{cfg.dataset.batch_size}_ep{cfg.trainer.max_epochs}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, ckpt_dir / "config.yaml")

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val/bacc",
            mode="max",
            save_top_k=1,
            filename="best",
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/bacc",
            mode="max",
            patience=cfg.trainer.early_stopping_patience,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        min_epochs=cfg.trainer.min_epochs,
        accelerator=cfg.trainer.accelerator,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        logger=wandb_logger,
        log_every_n_steps=10,
        callbacks=callbacks,
    )

    trainer.fit(module, dm)
    trainer.test(module, dm, ckpt_path="best")


if __name__ == "__main__":
    train()

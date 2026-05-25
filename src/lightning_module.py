import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics.functional import accuracy, precision, recall, f1_score, auroc
import wandb
from utils.metrics import plot_confusion_matrix


class VitLitModule(pl.LightningModule):
    def __init__(self, model: nn.Module, cfg):
        super().__init__()
        self.model = model
        self.lr = cfg.trainer.lr
        self.max_epochs = cfg.trainer.max_epochs
        self.warmup_epochs = cfg.trainer.warmup_epochs
        self.num_classes = cfg.model.num_classes
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=["model"])

        self._test_preds = []
        self._test_targets = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch):
        x, y = batch
        y = y.squeeze(1).long()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probs = logits.softmax(dim=1)
        preds = probs.argmax(dim=1)

        kwargs_macro = dict(task="multiclass", num_classes=self.num_classes, average="macro")
        kwargs_micro = dict(task="multiclass", num_classes=self.num_classes, average="micro")

        acc    = accuracy( preds, y, **kwargs_micro)
        prec   = precision(preds, y, **kwargs_macro)
        rec    = recall(   preds, y, **kwargs_macro)
        f1     = f1_score( preds, y, **kwargs_macro)
        auc    = auroc(    probs, y, **kwargs_macro)

        return loss, acc, prec, rec, f1, auc, preds, y

    def training_step(self, batch, batch_idx):
        loss, acc, prec, rec, f1, auc, _, _ = self._shared_step(batch)
        self.log_dict({
            "train/loss": loss,
            "train/acc":  acc,
            "train/prec": prec,
            "train/rec":  rec,
            "train/f1":   f1,
            "train/auc":  auc,
        }, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, prec, rec, f1, auc, _, _ = self._shared_step(batch)
        self.log_dict({
            "val/loss": loss,
            "val/acc":  acc,
            "val/prec": prec,
            "val/rec":  rec,
            "val/f1":   f1,
            "val/auc":  auc,
        }, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        _, acc, prec, rec, f1, auc, preds, targets = self._shared_step(batch)
        self.log_dict({
            "test/acc":  acc,
            "test/prec": prec,
            "test/rec":  rec,
            "test/f1":   f1,
            "test/auc":  auc,
        }, sync_dist=True)
        self._test_preds.append(preds.cpu())
        self._test_targets.append(targets.cpu())

    def on_test_epoch_end(self):
        preds = torch.cat(self._test_preds).numpy()
        targets = torch.cat(self._test_targets).numpy()
        self._test_preds.clear()
        self._test_targets.clear()

        fig = plot_confusion_matrix(preds, targets, self.num_classes)
        if self.logger is not None:
            self.logger.experiment.log({"test/confusion_matrix": wandb.Image(fig)})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        warmup = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=self.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs - self.warmup_epochs,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

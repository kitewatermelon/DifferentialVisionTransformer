# PathMNIST Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PathMNIST 9-class 분류 학습 파이프라인 구축 (Hydra config groups + PyTorch Lightning + WandB)

**Architecture:** Hydra config groups로 모델 선택(`python src/train.py model=diff_vit`). `PathMNISTDataModule`(LightningDataModule)이 medmnist 공식 split을 제공하고, `VitLitModule`(LightningModule)이 학습/검증/테스트 로직을 담당. WandbLogger로 전 메트릭 로깅.

**Tech Stack:** Python, PyTorch, timm 1.0.27, pytorch-lightning, hydra-core 1.3.2, wandb 0.27.0, medmnist 3.0.2

**실행 환경:** 반드시 `conda activate diffvit` 후 프로젝트 루트(`/home/dministrator/DifferentialVisionTransformer`)에서 실행

---

## File Map

| 파일 | 역할 |
|------|------|
| `src/configs/config.yaml` | Hydra 메인 config (defaults 선언) |
| `src/configs/model/vit.yaml` | ViT 모델 config |
| `src/configs/model/diff_vit.yaml` | DiffViT (전체) config |
| `src/configs/model/a6d6.yaml` | A6D6 (후반 6층 diff) config |
| `src/configs/model/odd_diff_vit.yaml` | 홀수 층 diff config |
| `src/configs/model/even_diff_vit.yaml` | 짝수 층 diff config |
| `src/configs/dataset/pathmnist.yaml` | 데이터셋 config |
| `src/configs/trainer/default.yaml` | 학습 하이퍼파라미터 |
| `src/configs/wandb/default.yaml` | WandB 설정 |
| `src/__init__.py` | src 패키지 init |
| `src/model/__init__.py` | model 패키지 init |
| `src/dataset/__init__.py` | dataset 패키지 init |
| `src/model/builder.py` | config.name → nn.Module 팩토리 |
| `src/dataset/medmnist.py` | PathMNISTDataModule (LightningDataModule) |
| `src/lightning_module.py` | VitLitModule (LightningModule) |
| `src/train.py` | Hydra 엔트리포인트 |
| `tests/conftest.py` | pytest sys.path 설정 (src/ 를 path에 추가) |
| `tests/test_builder.py` | builder 단위 테스트 |
| `tests/test_lightning_module.py` | VitLitModule 단위 테스트 |
| `tests/test_datamodule.py` | PathMNISTDataModule 단위 테스트 |

---

### Task 1: 의존성 확인 및 패키지 구조 생성

**Files:**
- Create: `src/__init__.py`
- Create: `src/model/__init__.py`
- Create: `src/dataset/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: pytorch-lightning 설치 확인**

```bash
conda activate diffvit
pip show pytorch-lightning
```

설치되어 있지 않으면:
```bash
pip install pytorch-lightning
```

Expected: `Name: pytorch-lightning` 출력

- [ ] **Step 2: 빈 __init__.py 파일 생성**

```bash
touch src/__init__.py src/model/__init__.py src/dataset/__init__.py
mkdir -p tests && touch tests/__init__.py
```

- [ ] **Step 3: tests/conftest.py 작성**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

- [ ] **Step 4: Commit**

```bash
git add src/__init__.py src/model/__init__.py src/dataset/__init__.py tests/
git commit -m "chore: add package init files and test conftest"
```

---

### Task 2: Hydra config 파일 작성

**Files:**
- Create: `src/configs/config.yaml`
- Create: `src/configs/model/vit.yaml`
- Create: `src/configs/model/diff_vit.yaml`
- Create: `src/configs/model/a6d6.yaml`
- Create: `src/configs/model/odd_diff_vit.yaml`
- Create: `src/configs/model/even_diff_vit.yaml`
- Create: `src/configs/dataset/pathmnist.yaml`
- Create: `src/configs/trainer/default.yaml`
- Create: `src/configs/wandb/default.yaml`

- [ ] **Step 1: config 디렉토리 구조 생성**

```bash
mkdir -p src/configs/model src/configs/dataset src/configs/trainer src/configs/wandb
```

- [ ] **Step 2: src/configs/config.yaml 작성**

```yaml
defaults:
  - model: vit
  - dataset: pathmnist
  - trainer: default
  - wandb: default
  - _self_

seed: 42
```

- [ ] **Step 3: 모델 config 5개 작성**

`src/configs/model/vit.yaml`:
```yaml
name: vit
backbone: vit_tiny_patch16_224
num_classes: 9
```

`src/configs/model/diff_vit.yaml`:
```yaml
name: diff_vit
backbone: vit_tiny_patch16_224
num_classes: 9
```

`src/configs/model/a6d6.yaml`:
```yaml
name: a6d6
backbone: vit_tiny_patch16_224
num_classes: 9
```

`src/configs/model/odd_diff_vit.yaml`:
```yaml
name: odd_diff_vit
backbone: vit_tiny_patch16_224
num_classes: 9
```

`src/configs/model/even_diff_vit.yaml`:
```yaml
name: even_diff_vit
backbone: vit_tiny_patch16_224
num_classes: 9
```

- [ ] **Step 4: dataset/trainer/wandb config 작성**

`src/configs/dataset/pathmnist.yaml`:
```yaml
image_size: 224
batch_size: 128
num_workers: 4
```

`src/configs/trainer/default.yaml`:
```yaml
max_epochs: 100
lr: 1.0e-4
accelerator: gpu
precision: 16-mixed
```

`src/configs/wandb/default.yaml`:
```yaml
project: DifferentialVisionTransformer
entity: kitewatermelon
tags: []
name: ${model.name}-run
```

- [ ] **Step 5: config 로드 확인**

```bash
conda activate diffvit && cd /home/dministrator/DifferentialVisionTransformer
python -c "
import os
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
with initialize_config_dir(config_dir=os.path.abspath('src/configs'), version_base=None):
    cfg = compose(config_name='config')
    print(OmegaConf.to_yaml(cfg))
"
```

Expected:
```
model:
  name: vit
  backbone: vit_tiny_patch16_224
  num_classes: 9
dataset:
  image_size: 224
  batch_size: 128
  num_workers: 4
...
```

- [ ] **Step 6: Commit**

```bash
git add src/configs/
git commit -m "feat: add Hydra config group files"
```

---

### Task 3: model/builder.py (TDD)

**Files:**
- Create: `src/model/builder.py`
- Create: `tests/test_builder.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_builder.py`:
```python
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
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

```bash
conda activate diffvit && cd /home/dministrator/DifferentialVisionTransformer
pytest tests/test_builder.py -v
```

Expected: `ModuleNotFoundError: No module named 'model.builder'`

- [ ] **Step 3: src/model/builder.py 작성**

```python
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
```

- [ ] **Step 4: 테스트 실행 (통과 확인)**

```bash
conda activate diffvit && cd /home/dministrator/DifferentialVisionTransformer
pytest tests/test_builder.py -v
```

Expected: 6개 테스트 전부 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/model/builder.py tests/test_builder.py
git commit -m "feat: add model builder factory with tests"
```

---

### Task 4: dataset/medmnist.py — PathMNISTDataModule (TDD)

**Files:**
- Modify: `src/dataset/medmnist.py`
- Create: `tests/test_datamodule.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_datamodule.py`:
```python
import pytest
from omegaconf import OmegaConf


@pytest.fixture
def cfg():
    return OmegaConf.create({
        "dataset": {
            "image_size": 224,
            "batch_size": 4,
            "num_workers": 0,
        }
    })


def test_datamodule_instantiates(cfg):
    from dataset.medmnist import PathMNISTDataModule
    dm = PathMNISTDataModule(cfg)
    assert dm.batch_size == 4
    assert dm.image_size == 224
    assert dm.num_workers == 0


@pytest.mark.slow
def test_datamodule_dataloaders_shape(cfg):
    import torch
    from dataset.medmnist import PathMNISTDataModule
    dm = PathMNISTDataModule(cfg)
    dm.setup()
    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == (4, 3, 224, 224)
    assert y.shape[0] == 4
    assert x.dtype == torch.float32
```

(느린 테스트는 `pytest -m "not slow"` 로 제외 가능)

- [ ] **Step 2: 테스트 실행 (실패 확인)**

```bash
conda activate diffvit && cd /home/dministrator/DifferentialVisionTransformer
pytest tests/test_datamodule.py::test_datamodule_instantiates -v
```

Expected: `ModuleNotFoundError` 또는 import error

- [ ] **Step 3: src/dataset/medmnist.py 작성**

```python
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
```

- [ ] **Step 4: 테스트 실행 (통과 확인)**

```bash
conda activate diffvit && cd /home/dministrator/DifferentialVisionTransformer
pytest tests/test_datamodule.py::test_datamodule_instantiates -v
```

Expected: PASSED

- [ ] **Step 5: Commit**

```bash
git add src/dataset/medmnist.py tests/test_datamodule.py
git commit -m "feat: add PathMNISTDataModule with tests"
```

---

### Task 5: lightning_module.py — VitLitModule (TDD)

**Files:**
- Create: `src/lightning_module.py`
- Create: `tests/test_lightning_module.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_lightning_module.py`:
```python
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
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

```bash
conda activate diffvit && cd /home/dministrator/DifferentialVisionTransformer
pytest tests/test_lightning_module.py -v
```

Expected: `ModuleNotFoundError: No module named 'lightning_module'`

- [ ] **Step 3: src/lightning_module.py 작성**

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl


class VitLitModule(pl.LightningModule):
    def __init__(self, model: nn.Module, cfg):
        super().__init__()
        self.model = model
        self.lr = cfg.trainer.lr
        self.max_epochs = cfg.trainer.max_epochs
        self.num_classes = cfg.model.num_classes
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch):
        x, y = batch
        y = y.squeeze(1).long()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        _, acc = self._shared_step(batch)
        self.log("test_acc", acc, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
```

- [ ] **Step 4: 테스트 실행 (통과 확인)**

```bash
conda activate diffvit && cd /home/dministrator/DifferentialVisionTransformer
pytest tests/test_lightning_module.py -v
```

Expected: 3개 테스트 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/lightning_module.py tests/test_lightning_module.py
git commit -m "feat: add VitLitModule with training/val/test steps"
```

---

### Task 6: train.py — Hydra 엔트리포인트

**Files:**
- Modify: `src/train.py`

- [ ] **Step 1: src/train.py 작성**

```python
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
```

- [ ] **Step 2: config dry-run 확인 (학습 없이 config만 확인)**

```bash
conda activate diffvit && cd /home/dministrator/DifferentialVisionTransformer
python src/train.py --cfg job
```

Expected: config 내용 출력, 에러 없음

- [ ] **Step 3: Commit**

```bash
git add src/train.py
git commit -m "feat: add Hydra train entrypoint with WandB logger"
```

---

### Task 7: 전체 테스트 + 스모크 테스트

- [ ] **Step 1: 전체 단위 테스트 실행 (느린 테스트 제외)**

```bash
conda activate diffvit && cd /home/dministrator/DifferentialVisionTransformer
pytest tests/ -v -m "not slow"
```

Expected: 10개 이상 테스트 PASSED, 0 FAILED

- [ ] **Step 2: vit 스모크 트레인 (1 epoch, CPU, WandB 비활성)**

```bash
conda activate diffvit && cd /home/dministrator/DifferentialVisionTransformer
python src/train.py \
  trainer.max_epochs=1 \
  trainer.accelerator=cpu \
  trainer.precision=32 \
  wandb.tags="[smoke-test]"
```

Expected: 1 epoch 완료, `val_loss`/`val_acc` 출력, WandB run 생성

- [ ] **Step 3: diff_vit 모델 스위칭 확인**

```bash
conda activate diffvit && cd /home/dministrator/DifferentialVisionTransformer
python src/train.py \
  model=diff_vit \
  trainer.max_epochs=1 \
  trainer.accelerator=cpu \
  trainer.precision=32
```

Expected: diff_vit 모델로 1 epoch 완료

- [ ] **Step 4: 최종 커밋**

```bash
git add -A
git commit -m "test: smoke test passed for vit and diff_vit"
```

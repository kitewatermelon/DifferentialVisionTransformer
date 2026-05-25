# PathMNIST Training Pipeline Design

## Overview

PathMNIST (9-class colon pathology classification, 224×224) 학습 파이프라인.
Hydra config groups + PyTorch Lightning + WandB 조합으로 구성.
모델은 vit / diff_vit / a6d6 / odd_diff_vit / even_diff_vit 중 CLI 한 줄로 선택.

## Architecture

```
src/
├── configs/
│   ├── config.yaml              # 메인: defaults 선언
│   ├── model/
│   │   ├── vit.yaml
│   │   ├── diff_vit.yaml
│   │   ├── a6d6.yaml
│   │   ├── odd_diff_vit.yaml
│   │   └── even_diff_vit.yaml
│   ├── dataset/
│   │   └── pathmnist.yaml
│   ├── trainer/
│   │   └── default.yaml
│   └── wandb/
│       └── default.yaml
├── model/
│   ├── vit.py                   # 기존
│   ├── diff_vit.py              # 기존
│   └── builder.py               # config → model 팩토리 (신규)
├── dataset/
│   └── medmnist.py              # PathMNISTDataModule (신규)
├── lightning_module.py          # VitLitModule (신규)
└── train.py                     # Hydra 엔트리포인트 (신규)
```

## Config Structure

### `configs/config.yaml`
```yaml
defaults:
  - model: vit
  - dataset: pathmnist
  - trainer: default
  - wandb: default
  - _self_
```

### `configs/model/*.yaml` (예: diff_vit.yaml)
```yaml
name: diff_vit
backbone: vit_tiny_patch16_224
num_classes: 9
```

모든 모델 yaml의 필드: `name`, `backbone`, `num_classes`.
기본 backbone은 `vit_tiny_patch16_224`.

### `configs/dataset/pathmnist.yaml`
```yaml
image_size: 224
batch_size: 128
num_workers: 4
```

medmnist 라이브러리의 공식 train/val/test split 사용 (자동 다운로드 지원).

### `configs/trainer/default.yaml`
```yaml
max_epochs: 100
lr: 1e-4
accelerator: gpu
precision: 16-mixed
```

### `configs/wandb/default.yaml`
```yaml
project: DifferentialVisionTransformer
entity: kitewatermelon
tags: []
name: ${model.name}-${now:%Y%m%d_%H%M%S}
```

project, entity, tags, run name 전부 Hydra에서 관리.

## Components

### `model/builder.py`
- `build_model(cfg) -> nn.Module`
- cfg.name 기준으로 팩토리 딕셔너리에서 빌더 선택
- `model.reset_classifier(cfg.num_classes)` 로 분류 헤드 교체

### `dataset/medmnist.py` — `PathMNISTDataModule(LightningDataModule)`
- `medmnist.PathMNIST` 사용 (as_rgb=True, size=224)
- train/val/test DataLoader 반환
- transforms: train은 RandomHorizontalFlip + Normalize, val/test는 Normalize만

### `lightning_module.py` — `VitLitModule(LightningModule)`
- Loss: CrossEntropyLoss
- Optimizer: AdamW (lr from cfg.trainer.lr)
- Scheduler: CosineAnnealingLR (T_max=max_epochs)
- 매 step마다 train_loss, val_loss, val_acc, test_acc WandB 로깅

### `train.py`
- `@hydra.main(config_path="configs", config_name="config", version_base=None)`
- WandbLogger 초기화 → Trainer 생성 → fit

## Usage

```bash
conda activate diffvit

# 기본 (vit)
python src/train.py

# 모델 변경
python src/train.py model=diff_vit
python src/train.py model=a6d6

# 하이퍼파라미터 오버라이드
python src/train.py model=diff_vit trainer.max_epochs=50 wandb.tags="[exp1]"

# WandB run name 직접 지정
python src/train.py wandb.name=my-run-01
```

## Data Flow

```
medmnist.PathMNIST → PathMNISTDataModule → DataLoader
                                               ↓
                                        VitLitModule.training_step
                                               ↓
                                        build_model(cfg.model)
                                               ↓
                                        WandbLogger → WandB
```

## Key Decisions

- **medmnist 라이브러리 사용**: 공식 split 보장, npz 직접 파싱 불필요
- **Hydra config groups**: 모델 스위칭이 CLI 한 줄로 가능
- **precision=16-mixed**: A100/V100 환경 기준 속도 향상
- **CosineAnnealingLR**: PathMNIST 규모에서 안정적 수렴

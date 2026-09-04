# A6D6 — Differential Vision Transformer

Differential Attention block-mixing 기반 의료 영상 분류 연구.

- **Experiment 1**: MedMNIST CT subsets — corruption robustness (순수 아키텍처 효과 측정)
- **Experiment 2**: Retinal fundus (IDRiD, EyePACS) — structural distracter 속 discriminative patch 탐지 능력

## Setup

```bash
conda create -n a6d6 python=3.10 -y
conda activate a6d6
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
wandb login
```

## Models

| 모델 | 구성 | MedMNIST (patch 8) | Fundus (patch 16, 512×512) |
|---|---|---|---|
| ViT-Ti | 표준 self-attention ×12 | `model=vit` | `model=vit_p16` |
| ViDT-Ti | Differential Attention ×12 | `model=diff_vit` | `model=diff_vit_p16` |
| A6D6-Ti | 표준 ×6 + Differential ×6 | `model=a6d6` | `model=a6d6_p16` |

## Datasets

| Dataset | Classes | Size | Hydra key |
|---|---|---|---|
| OrganAMNIST | 11 | ~58K | `dataset=organamnist` |
| OrganCMNIST | 11 | ~23K | `dataset=organcmnist` |
| OrganSMNIST | 11 | ~25K | `dataset=organsmnist` |
| IDRiD | 5 (DR 0–4) | 516 | `dataset=idrid` |
| EyePACS | 5 (DR 0–4) | 35K | `dataset=eyepacs` |

## Training

```bash
# MedMNIST (데이터 자동 다운로드)
CUDA_VISIBLE_DEVICES=0 bash run_vit.sh        # ViT × OA/OC/OS
CUDA_VISIBLE_DEVICES=0 bash run_diff_vit.sh   # ViDT × OA/OC/OS
CUDA_VISIBLE_DEVICES=0 bash run_a6d6.sh       # A6D6 × OA/OC/OS

# IDRiD
CUDA_VISIBLE_DEVICES=1 bash run_idrid.sh      # ViT/A6D6/DiffViT × IDRiD

# EyePACS
CUDA_VISIBLE_DEVICES=1 bash run_eyepacs.sh    # ViT/A6D6/DiffViT × EyePACS

# 단일 실험
python src/train.py dataset=organamnist model=a6d6
python src/train.py dataset=idrid model=a6d6_p16 wandb.project=A6D6_idrid
```

## Testing

```bash
python src/test.py dataset=organamnist model=a6d6
```

## Wandb

| Project | 실험 |
|---|---|
| [`A6D6`](https://wandb.ai/DiffViT/A6D6) | MedMNIST CT |
| [`A6D6_idrid`](https://wandb.ai/DiffViT/A6D6_idrid) | IDRiD |
| [`A6D6_eyepacs`](https://wandb.ai/DiffViT/A6D6_eyepacs) | EyePACS |

## Project Map

전체 파일 구조 및 코드 흐름: [docs/PROJECT_MAP.md](docs/PROJECT_MAP.md)

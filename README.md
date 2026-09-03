# A6D6 — Differential Vision Transformer

Differential Attention block-mixing 기반 CT 영상 분류 robustness 연구 (MedMNIST+ CT subsets).

## Setup

```bash
conda create -n a6d6 python=3.10 -y
conda activate a6d6
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
wandb login
```

## Training

```bash
# 모델별 전체 실험 (OA/OC/OS × 3 datasets)
./run_vit.sh        # ViT-Ti/8
./run_diff_vit.sh   # ViDT-Ti/8
./run_a6d6.sh       # A6D6-Ti/8

# 단일 실험
python src/train.py dataset=organamnist model=a6d6
```

## Testing

```bash
python src/test.py dataset=organamnist model=a6d6
```

## Models

| 모델 | 구성 | Hydra key |
|---|---|---|
| ViT-Ti/8 | 표준 self-attention ×12 | `model=vit` |
| ViDT-Ti/8 | Differential Attention ×12 | `model=diff_vit` |
| A6D6-Ti/8 | 표준 ×6 + Differential ×6 | `model=a6d6` |

## Datasets

| Dataset | Classes | Hydra key |
|---|---|---|
| OrganAMNIST | 11 | `dataset=organamnist` |
| OrganCMNIST | 11 | `dataset=organcmnist` |
| OrganSMNIST | 11 | `dataset=organsmnist` |

## Wandb

- Project: [DifferentialVisionTransformer](https://wandb.ai/DiffViT/DifferentialVisionTransformer)

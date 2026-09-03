# A6D6 Project Map

Differential Attention block-mixing 기반 CT 영상 분류 robustness 연구.

## 디렉토리 구조

```
DifferentialVisionTransformer/
├── src/
│   ├── train.py                    # 학습 진입점 (Hydra CLI)
│   ├── test.py                     # 평가 + attention 시각화
│   ├── lightning_module.py         # VitLitModule (학습/평가 로직)
│   ├── model/
│   │   ├── builder.py              # build_model() — 모델 팩토리
│   │   ├── vit.py                  # 표준 ViT (timm)
│   │   └── diff_vit.py             # DiffAttn 교체 로직, A6D6/ViDT 등
│   ├── dataset/
│   │   └── medmnist.py             # MedMNISTDataModule (범용)
│   ├── utils/
│   │   ├── vis_attn.py             # Layer×Head CLS attention heatmap
│   │   └── metrics.py              # confusion matrix 시각화
│   └── configs/
│       ├── config.yaml             # Hydra root config
│       ├── model/                  # vit, diff_vit, a6d6, odd_diff_vit, even_diff_vit
│       ├── dataset/                # organamnist, organcmnist, organsmnist, pathmnist
│       ├── trainer/default.yaml
│       └── wandb/default.yaml
├── data/                           # MedMNIST 데이터 (auto-download)
├── outputs/checkpoints/            # 학습 체크포인트 저장
│   └── {dataset}/{model}/lr..._bs..._ep.../
│       ├── best.ckpt
│       ├── last.ckpt
│       └── config.yaml
├── docs/
│   └── attn_maps/                  # attention 시각화 결과물
├── run_vit.sh                      # ViT 전담 실행
├── run_diff_vit.sh                 # ViDT 전담 실행
└── run_a6d6.sh                     # A6D6 전담 실행
```

## 데이터 파이프라인

- **데이터셋**: MedMNIST+ CT subsets (224×224, `as_rgb=True`)
  - OrganAMNIST (11 classes, Axial)
  - OrganCMNIST (11 classes, Coronal)
  - OrganSMNIST (11 classes, Sagittal)
- **전처리**: `ToTensor → RandomHorizontalFlip(train만) → Normalize(0.5, 0.5)`
- **Corruption augmentation 없음** — clean 데이터만으로 학습

## 모델

| 모델 | 설명 | Hydra key |
|---|---|---|
| ViT-Ti/8 | 표준 self-attention ×12 | `model=vit` |
| ViDT-Ti/8 | Differential Attention ×12 | `model=diff_vit` |
| A6D6-Ti/8 | 표준 ×6 + Differential ×6 | `model=a6d6` |

공통: `vit_tiny_patch16_224` backbone + `patch_size=8` override, embed_dim=192, depth=12, heads=3.

## 학습 설정

| 항목 | 값 |
|---|---|
| Batch size | 256 |
| Optimizer | AdamW (lr=1e-4, wd=0.05) |
| LR schedule | 5-epoch linear warmup → cosine |
| Epochs | max 300, min 40 |
| Early stopping | val/bacc 기준, patience 15 |
| Seed | 42 |
| Precision | 16-mixed |

## 재현 방법

```bash
# 환경 설정
conda activate a6d6  # or vit, diff_vit — 환경 이름은 자유
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 단일 실험
cd src
python train.py dataset=organamnist model=a6d6

# 모델별 전체 실험
./run_vit.sh       # ViT: OA, OC, OS
./run_diff_vit.sh  # ViDT: OA, OC, OS
./run_a6d6.sh      # A6D6: OA, OC, OS

# 테스트 (학습 완료 후)
cd src
python test.py dataset=organamnist model=a6d6
```

## Wandb

- **Project**: [DifferentialVisionTransformer](https://wandb.ai/DiffViT/DifferentialVisionTransformer)
- **Run naming**: `{dataset}_{model}_seed42` (예: `organamnist_a6d6_seed42`)
- **로깅**: loss, acc, bacc, prec, f1, auc, lr, confusion matrix, attention maps

## 학습 결과 요약

> ⚠️ 실험 완료 후 아래 표를 채워 주세요.

### Clean bACC

| Dataset | ViT-Ti/8 | ViDT-Ti/8 | A6D6-Ti/8 |
|---|---|---|---|
| OrganAMNIST | — | — | — |
| OrganCMNIST | — | — | — |
| OrganSMNIST | — | — | — |

### Robustness (AEI ↓)

| Dataset | ViT-Ti/8 | ViDT-Ti/8 | A6D6-Ti/8 |
|---|---|---|---|
| OrganAMNIST | — | — | — |
| OrganCMNIST | — | — | — |
| OrganSMNIST | — | — | — |

### Calibration (ECE ↓)

| Dataset | ViT-Ti/8 | ViDT-Ti/8 | A6D6-Ti/8 |
|---|---|---|---|
| OrganAMNIST | — | — | — |
| OrganCMNIST | — | — | — |
| OrganSMNIST | — | — | — |

## Attention 시각화

- 경로: `docs/attn_maps/{model}/L{layer}_H{head}.png`
- Wandb: 각 run의 `attn/cls_attention` 패널
- Clean vs corrupted 비교, A6D6 앞 6블록(std) / 뒤 6블록(diff) 비교 figure

> ⚠️ 시각화 결과물은 `test.py` 실행 후 wandb에 업로드됩니다.

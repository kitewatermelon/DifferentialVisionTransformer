# A6D6

Differential Attention block-mixing 기반 의료 영상 분류 연구.

## 실험 축

1. **MedMNIST CT — Corruption Robustness**: MedMNIST+ CT subsets에서 corruption에 대한 아키텍처 효과 측정
2. **Retinal Fundus — Discriminative Patch 탐지**: IDRiD/EyePACS에서 structural distracter(혈관, optic disc) 속 DR 병변 선별 능력 검증

## Dataset

### MedMNIST CT (Experiment 1)
- `./data/data/`에 위치 (auto-download)
  - OrganAMNIST+ (OA), OrganCMNIST+ (OC), OrganSMNIST+ (OS)
- 224×224, patch_size=8
- Corruption 평가는 MedMNIST-C 벤치마크(Noise / Blur / Digital 계열, severity 1~5) 사용
- 학습 시 corruption 관련 augmentation(noise/blur 시뮬레이션)은 사용하지 않음 — 순수 clean 데이터로만 학습

### Retinal Fundus (Experiment 2)
- IDRiD: `./data/idrid/` (Kaggle: `aaryapatel98/indian-diabetic-retinopathy-image-dataset`)
  - 5-class DR grading (Grade 0–4), train 371+val 42 / test 103 (공식 split)
  - Pixel-level segmentation mask 제공 (81장) — attention localization 검증 가능
- EyePACS: `./data/eyepacs/` (HuggingFace: `ctmedtech/EYEPACS`)
  - 5-class DR grading (Grade 0–4), 35,108장, 8:1:1 random split
  - `trainLabels.csv` (columns: image, level), 이미지: `eyepacs_preprocess/eyepacs_preprocess/*.jpeg`
- 512×512, patch_size=16

## Models
세 가지 아키텍처를 동일한 ViT-Ti backbone에서 비교:

| 모델 | 구성 | MedMNIST (patch 8) | Fundus (patch 16) |
|---|---|---|---|
| ViT-Ti | 표준 self-attention만 사용 (L×12) | `model=vit` | `model=vit_p16` |
| ViDT-Ti | Differential Attention만 사용 (L×12) | `model=diff_vit` | `model=diff_vit_p16` |
| A6D6-Ti | 표준 attention(앞 6블록) + Differential Attention(뒤 6블록) | `model=a6d6` | `model=a6d6_p16` |

- Differential Attention은 `[softmax(Q1K1^T/√d) − λ·softmax(Q2K2^T/√d)]V` 형태
- λ_init = 0.8 − 0.6 × exp(−0.3(l−1)), layer index l 기준, 레이어 내 헤드 간 공유
- 공통 backbone: `vit_tiny_patch16_224` (embed_dim=192, depth=12, heads=3), `pretrained=False`

## Training Config
- Batch size: MedMNIST 256 (A6D6: 64), Fundus 16
- Optimizer: AdamW, lr 1e-4, weight decay 0.05
- LR schedule: 5-epoch linear warmup → cosine
- Epochs: max 300, min 40 (min 이후 early stopping 적용, patience 15), best checkpoint는 clean validation bACC 기준으로 저장
- Precision: 16-mixed
- Seed: 42

## Augmentation
- Random Resized Crop: scale [0.75, 1.0], ratio [0.95, 1.05]
- Random Rotation: ±15°
- Color Jitter: brightness 0.4, contrast 0.4, saturation 0.4, hue 0.1
- Random Horizontal Flip (fundus 데이터셋)
- **Noise/Blur 시뮬레이션 augmentation은 사용하지 않음** — corruption robustness를 순수 아키텍처 효과로 측정하기 위함 (corruption은 test-time에만 적용)

## Evaluation
- 주 지표: balanced accuracy (bACC) — class imbalance 대응
- Robustness 지표: AEI (Absolute Error Increase) = BE_corrupted − BE_clean (MedMNIST만)
- Calibration 지표: ECE, NLL, Brier score (clean vs corrupted, Δ 비교)

## 코드 구조 핵심
- `src/train.py`: Hydra 진입점, dataset name으로 DataModule 자동 선택 (medmnist/idrid/eyepacs)
- `src/model/builder.py`: config의 extra keys(patch_size, img_size 등)를 timm에 전달
- `src/model/diff_vit.py`: `replace_attention_with_diff()` — 지정 블록의 attention을 DiffAttention으로 교체
- `src/lightning_module.py`: `VitLitModule` — CrossEntropyLoss, multiclass metrics
- `src/dataset/`: medmnist.py, idrid.py, eyepacs.py — 각각 독립 DataModule

## Experiment Tracking (wandb)
- Entity: DiffViT
- Projects:
  - `A6D6`: MedMNIST CT 실험
  - `A6D6_idrid`: IDRiD 실험
  - `A6D6_eyepacs`: EyePACS 실험
- Run naming: `{dataset}_{model}_seed{seed}`
- 로깅: train/val loss, acc, bacc, prec, f1, auc, lr, confusion matrix, attention maps

## Run Scripts
- `run_vit.sh` / `run_diff_vit.sh` / `run_a6d6.sh`: MedMNIST (OA/OC/OS)
- `run_idrid.sh`: IDRiD (ViT/A6D6/DiffViT, wandb: A6D6_idrid)
- `run_eyepacs.sh`: EyePACS 3모델 통합 (wandb: A6D6_eyepacs)
- `run_vit_eyepacs.sh` / `run_diff_vit_eyepacs.sh` / `run_a6d6_eyepacs.sh`: EyePACS 개별

## Output
- 체크포인트: `outputs/checkpoints/{dataset}/{model}/lr{lr}_bs{bs}_ep{ep}/`
- 프로젝트 맵: `docs/PROJECT_MAP.md`

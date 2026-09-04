# A6D6 Project Map

Differential Attention block-mixing 기반 의료 영상 분류 연구.
두 축으로 실험: (1) MedMNIST CT — corruption robustness, (2) Retinal fundus (IDRiD, EyePACS) — structural distracter 속 discriminative patch 탐지.

## 디렉토리 구조

```
DifferentialVisionTransformer/
├── src/
│   ├── train.py                    # 학습 진입점 (Hydra CLI), 데이터셋별 DataModule 자동 선택
│   ├── test.py                     # 평가 + attention 시각화 (MedMNIST 전용, 확장 필요)
│   ├── lightning_module.py         # VitLitModule (학습/평가 로직, CrossEntropy, bacc 모니터링)
│   ├── model/
│   │   ├── builder.py              # build_model() — 모델 팩토리, config의 extra keys를 timm에 전달
│   │   ├── vit.py                  # 표준 ViT (timm.create_model)
│   │   └── diff_vit.py             # DiffAttn 교체 로직 (replace_attention_with_diff)
│   │                               #   get_vit / get_diff_vit / get_a6d6 / get_odd_diff_vit / get_even_diff_vit
│   ├── dataset/
│   │   ├── medmnist.py             # MedMNISTDataModule — MedMNIST+ 범용
│   │   ├── idrid.py                # IDRiDDataModule — 공식 train/test split, train에서 9:1로 val 분리
│   │   └── eyepacs.py              # EyePACSDataModule — 35K장, 8:1:1 random split
│   ├── utils/
│   │   ├── vis_attn.py             # Layer×Head CLS attention heatmap 시각화
│   │   └── metrics.py              # confusion matrix 시각화
│   └── configs/
│       ├── config.yaml             # Hydra root config (defaults, seed=42)
│       ├── model/
│       │   ├── vit.yaml            # ViT-Ti, patch_size=8 (MedMNIST용)
│       │   ├── diff_vit.yaml       # ViDT-Ti, patch_size=8
│       │   ├── a6d6.yaml           # A6D6-Ti, patch_size=8
│       │   ├── vit_p16.yaml        # ViT-Ti, patch_size=16, img_size=512 (fundus용)
│       │   ├── diff_vit_p16.yaml   # ViDT-Ti, patch_size=16, img_size=512
│       │   ├── a6d6_p16.yaml       # A6D6-Ti, patch_size=16, img_size=512
│       │   ├── odd_diff_vit.yaml   # 홀수 블록만 DiffAttn
│       │   └── even_diff_vit.yaml  # 짝수 블록만 DiffAttn
│       ├── dataset/
│       │   ├── organamnist.yaml    # OA, 11 classes, 224×224, bs=256
│       │   ├── organcmnist.yaml    # OC, 11 classes, 224×224, bs=256
│       │   ├── organsmnist.yaml    # OS, 11 classes, 224×224, bs=256
│       │   ├── idrid.yaml          # IDRiD, 5 classes, 512×512, bs=16
│       │   ├── eyepacs.yaml        # EyePACS, 5 classes, 512×512, bs=16
│       │   └── pathmnist.yaml      # (미사용)
│       ├── trainer/default.yaml    # AdamW lr=1e-4, wd=0.05, cosine, max 300ep
│       └── wandb/default.yaml      # project=A6D6, entity=DiffViT
│
├── data/
│   ├── data/                       # MedMNIST 데이터 (auto-download)
│   ├── idrid/                      # IDRiD (kaggle download)
│   │   └── B.%20Disease%20Grading/ # Disease Grading subset
│   │       └── B. Disease Grading/
│   │           ├── 1. Original Images/{a. Training Set, b. Testing Set}/
│   │           └── 2. Groundtruths/{Training Labels.csv, Testing Labels.csv}
│   └── eyepacs/                    # EyePACS (HuggingFace download)
│       ├── trainLabels.csv         # image, level (0–4)
│       └── eyepacs_preprocess/eyepacs_preprocess/  # 35,108 .jpeg files
│
├── outputs/checkpoints/            # 학습 체크포인트
│   └── {dataset}/{model}/lr..._bs..._ep.../
│       ├── best.ckpt
│       ├── last.ckpt
│       └── config.yaml
│
├── run_vit.sh                      # ViT × MedMNIST (OA/OC/OS)
├── run_diff_vit.sh                 # ViDT × MedMNIST (OA/OC/OS)
├── run_a6d6.sh                     # A6D6 × MedMNIST (OA/OC/OS), bs=64
├── run_idrid.sh                    # ViT/A6D6/DiffViT × IDRiD (wandb: A6D6_idrid)
├── run_eyepacs.sh                  # ViT/A6D6/DiffViT × EyePACS (wandb: A6D6_eyepacs)
├── run_vit_eyepacs.sh              # ViT × EyePACS (단독)
├── run_diff_vit_eyepacs.sh         # DiffViT × EyePACS (단독)
├── run_a6d6_eyepacs.sh             # A6D6 × EyePACS (단독)
│
├── tests/                          # pytest 테스트
│   ├── test_builder.py
│   ├── test_datamodule.py
│   └── test_lightning_module.py
│
├── docs/
│   └── PROJECT_MAP.md              # 이 파일
├── CLAUDE.md                       # Claude Code 세션 컨텍스트
└── README.md
```

## 모델

| 모델 | 구성 | patch_size=8 (MedMNIST) | patch_size=16 (fundus) |
|---|---|---|---|
| ViT-Ti | 표준 self-attention ×12 | `model=vit` | `model=vit_p16` |
| ViDT-Ti | Differential Attention ×12 | `model=diff_vit` | `model=diff_vit_p16` |
| A6D6-Ti | 표준 ×6 + Differential ×6 | `model=a6d6` | `model=a6d6_p16` |

공통 backbone: `vit_tiny_patch16_224` (embed_dim=192, depth=12, heads=3), `pretrained=False`.

Differential Attention: `[softmax(Q1K1^T/√d) − λ·softmax(Q2K2^T/√d)]V`
- λ_init = 0.8 − 0.6 × exp(−0.3(l−1)), 레이어 내 헤드 간 공유

## 데이터셋

### Experiment 1: MedMNIST CT — Corruption Robustness

| Dataset | Classes | Size | Input | Hydra |
|---|---|---|---|---|
| OrganAMNIST | 11 | ~58K | 224×224, patch 8 | `dataset=organamnist` |
| OrganCMNIST | 11 | ~23K | 224×224, patch 8 | `dataset=organcmnist` |
| OrganSMNIST | 11 | ~25K | 224×224, patch 8 | `dataset=organsmnist` |

- Corruption 평가: MedMNIST-C (Noise/Blur/Digital, severity 1~5)
- 학습 시 corruption augmentation 없음

### Experiment 2: Retinal Fundus — Discriminative Patch 탐지

| Dataset | Classes | Train | Test | Input | Hydra |
|---|---|---|---|---|---|
| IDRiD | 5 (DR grade 0–4) | 371+42(val) | 103 (공식) | 512×512, patch 16 | `dataset=idrid` |
| EyePACS | 5 (DR grade 0–4) | ~28K | ~3.5K | 512×512, patch 16 | `dataset=eyepacs` |

- IDRiD: Kaggle `aaryapatel98/indian-diabetic-retinopathy-image-dataset`
- EyePACS: HuggingFace `ctmedtech/EYEPACS`
- 목적: structural distracter(혈관, optic disc) 속에서 미세동맥류 등 DR 병변을 선별적으로 attend하는 Diff Attention의 능력 검증

## 학습 설정

| 항목 | MedMNIST | Fundus (IDRiD/EyePACS) |
|---|---|---|
| Batch size | 256 (a6d6: 64) | 16 |
| Optimizer | AdamW (lr=1e-4, wd=0.05) | 동일 |
| LR schedule | 5-epoch warmup → cosine | 동일 |
| Epochs | max 300, min 40 | 동일 |
| Early stopping | val/bacc, patience 15 | 동일 |
| Precision | 16-mixed | 동일 |
| Seed | 42 | 동일 |

## Wandb 프로젝트

| 프로젝트 | 실험 |
|---|---|
| `A6D6` | MedMNIST CT subsets |
| `A6D6_idrid` | IDRiD fundus |
| `A6D6_eyepacs` | EyePACS fundus |

Entity: `DiffViT`, Run naming: `{dataset}_{model}_seed42`

## 핵심 코드 흐름

1. `train.py` → Hydra config 로드 → `build_model(cfg.model)` → DataModule 선택 → `VitLitModule` → `pl.Trainer.fit()`
2. `build_model()`: config의 `name` 키로 builder 선택 → timm 모델 생성 → (DiffAttn 모델이면) `replace_attention_with_diff()` → `reset_classifier(num_classes)`
3. `_shared_step()`: forward → CrossEntropyLoss + multiclass metrics (acc, bacc, prec, f1, auc)
4. Checkpoint: `outputs/checkpoints/{dataset}/{model}/lr..._bs..._ep.../best.ckpt`

## 재현 방법

```bash
# 환경 설정
conda activate a6d6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
wandb login

# MedMNIST 실험 (데이터 자동 다운로드)
CUDA_VISIBLE_DEVICES=0 bash run_vit.sh
CUDA_VISIBLE_DEVICES=0 bash run_a6d6.sh

# IDRiD (사전 다운로드 필요)
kaggle datasets download aaryapatel98/indian-diabetic-retinopathy-image-dataset -p data/idrid --unzip
CUDA_VISIBLE_DEVICES=1 bash run_idrid.sh

# EyePACS (사전 다운로드 필요, ~6.5GB)
# huggingface_hub로 data/eyepacs/에 다운로드 + 압축해제
CUDA_VISIBLE_DEVICES=1 bash run_eyepacs.sh
```

## 학습 결과 요약

> ⚠️ 실험 완료 후 아래 표를 채워 주세요.

### MedMNIST — Clean bACC

| Dataset | ViT-Ti/8 | ViDT-Ti/8 | A6D6-Ti/8 |
|---|---|---|---|
| OrganAMNIST | — | — | — |
| OrganCMNIST | — | — | — |
| OrganSMNIST | — | — | — |

### Fundus — Test bACC

| Dataset | ViT-Ti/16 | ViDT-Ti/16 | A6D6-Ti/16 |
|---|---|---|---|
| IDRiD | — | — | — |
| EyePACS | — | — | — |

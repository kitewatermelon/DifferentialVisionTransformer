# A6D6

Differential Attention block-mixing 기반 CT 영상 분류 robustness 연구 (MedMNIST+ CT subsets).

## Dataset
- `./data/`에 위치
  - OrganAMNIST+ (OA)
  - OrganCMNIST+ (OC)
  - OrganSMNIST+ (OS)
- Corruption 평가는 MedMNIST-C 벤치마크(Noise / Blur / Digital 계열, severity 1~5) 사용
- 학습 시 corruption 관련 augmentation(noise/blur 시뮬레이션)은 사용하지 않음 — 순수 clean 데이터로만 학습

## Models
세 가지 아키텍처를 동일한 ViT-Ti/8 backbone(patch size 8)에서 비교:

| 모델 | 구성 |
|---|---|
| ViT-Ti/8 | 표준 self-attention만 사용 (L×12) |
| ViDT-Ti/8 | Differential Attention만 사용 (L×12) |
| A6D6-Ti/8 | 표준 attention(앞 6블록) + Differential Attention(뒤 6블록) block-mixing |

- Differential Attention은 `[softmax(Q1K1^T/√d) − λ·softmax(Q2K2^T/√d)]V` 형태
- λ_init = 0.8 − 0.6 × exp(−0.3(l−1)), layer index l 기준, 레이어 내 헤드 간 공유

## Training Config
- Batch size: 256
- Optimizer: AdamW, lr 1e-4, weight decay 0.05
- LR schedule: cosine
- Epochs: max 300, min 40 (min 이후 early stopping 적용), best checkpoint는 clean validation bACC 기준으로 저장
- Seed: 42

## Augmentation
- Random Resized Crop: scale [0.75, 1.0], ratio [0.95, 1.05]
- Random Rotation: ±15°
- Color Jitter: brightness 0.4, contrast 0.4, saturation 0.4, hue 0.1
- **Noise/Blur 시뮬레이션 augmentation은 사용하지 않음** — corruption robustness를 순수 아키텍처 효과로 측정하기 위함 (corruption은 test-time에만 적용)
- Color Jitter는 색상/명암 관련 corruption(Color family)에 대한 노출을 만들기 때문에, Color 계열 corruption에 대해서는 강한 robustness 주장을 하지 않음 — 주요 robustness claim은 Noise/Blur/Digital 계열에 한정

## Evaluation
- 주 지표: balanced accuracy (bACC) — class imbalance 대응
- Robustness 지표: AEI (Absolute Error Increase) = BE_corrupted − BE_clean (BE = 1 − bACC), corruption family/severity 평균
- Calibration 지표: ECE, NLL, Brier score (clean vs corrupted, Δ 비교)

## Attention Visualization
- Layer × Head 별 CLS 토큰의 attention map 시각화
  - 대상: 표준 attention block은 softmax(QK^T/√d), Differential Attention block은 두 softmax 항의 차(diff map) 모두 저장
  - CLS → patch token attention weight를 patch grid(16×16 등 patch size 기준)로 reshape하여 heatmap으로 렌더링
  - 모델별(ViT-Ti/8, ViDT-Ti/8, A6D6-Ti/8) 전 레이어 × 전 헤드 grid 이미지로 저장 (예: `docs/attn_maps/{model}/L{layer}_H{head}.png`)
  - Clean 입력과 corrupted 입력(대표 severity 1개) 각각에 대해 생성하여 corruption 시 attention 변화 비교 가능하도록 구성
  - A6D6의 경우 앞 6블록(표준)과 뒤 6블록(differential)의 attention 패턴 차이를 한 화면에서 비교할 수 있는 요약 figure도 별도 생성

## Experiment Tracking (wandb)
- Project: https://wandb.ai/DiffViT
- 모든 학습 run은 wandb로 로깅
  - Project 단위로 OA/OC/OS × 3개 모델 run 구분 (예: run name `{dataset}_{model}_seed{seed}`)
  - 로깅 항목: train/val loss, clean bACC, corrupted bACC(family별), AEI, ECE/NLL/Brier(clean/corrupted), learning rate
  - Layer × Head별 attention heatmap을 학습 중 주기적으로(예: N epoch마다, 혹은 최종 checkpoint 기준) `wandb.Image`로 업로드
  - Config에 batch size, lr, wd, schedule, epoch, seed 등 학습 하이퍼파라미터 자동 기록

## Output
- 세 모델 학습/평가 완료 후 `docs/`에 project map 작성
  - 디렉토리 구조, 데이터 파이프라인, 모델별 학습 결과 요약, 재현 방법 포함
  - attention 시각화 결과물 경로 및 wandb project 링크 포함
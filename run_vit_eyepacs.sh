#!/bin/bash
cd "$(dirname "$0")"
for model in vit_p16; do
  echo "=== Training $model on EyePACS ==="
  python src/train.py dataset=eyepacs model=$model wandb.project=A6D6_eyepacs
done

#!/bin/bash
cd "$(dirname "$0")"
for model in vit_p16 a6d6_p16 diff_vit_p16; do
  echo "=== Training $model on IDRiD ==="
  python src/train.py dataset=idrid model=$model wandb.project=A6D6_idrid
done

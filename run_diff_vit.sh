#!/bin/bash
cd "$(dirname "$0")"
for ds in organamnist organcmnist organsmnist; do
  echo "=== Training diff_vit on $ds ==="
  python src/train.py dataset=$ds model=diff_vit dataset.batch_size=64
done

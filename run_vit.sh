#!/bin/bash
cd "$(dirname "$0")"
for ds in organamnist organcmnist organsmnist; do
  echo "=== Training vit on $ds ==="
  python src/train.py dataset=$ds model=vit
done

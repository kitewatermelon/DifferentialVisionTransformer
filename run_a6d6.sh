#!/bin/bash
cd "$(dirname "$0")"
for ds in organamnist organcmnist organsmnist; do
  echo "=== Training a6d6 on $ds ==="
  python src/train.py dataset=$ds model=a6d6
done

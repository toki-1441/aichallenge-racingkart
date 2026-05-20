#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python3 prepare_data.py synthetic \
  --out-train "$ROOT/datasets/synthetic/train" \
  --out-val "$ROOT/datasets/synthetic/val" \
  --num-train 512 --num-val 64 \
  --horizon 40 --num-heads 4 --seed 0

echo "[run_pipeline] starting train.py (Hydra)..."
python3 train.py

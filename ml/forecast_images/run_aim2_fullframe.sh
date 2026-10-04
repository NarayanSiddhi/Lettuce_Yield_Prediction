#!/usr/bin/env bash
# Aim 2: dense full-frame RAW pairs (+5d) → Pix2Pix for both trials
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-/data/home/sai/anaconda3/bin/python3}"
HORIZON="${HORIZON:-5}"
SIZE="${SIZE:-256}"
EPOCHS="${EPOCHS:-100}"
OUT="ml/forecast_images"
LOG="$OUT/fullframe_pipeline_h${HORIZON}.log"
mkdir -p "$OUT"
exec > >(tee "$LOG") 2>&1

echo "=== Full-frame Aim2 pipeline h=${HORIZON} size=${SIZE} === $(date -Is)"

echo ""
echo "=== [1/2] Build dense RAW pairs + preprocess ==="
$PY -u ml/forecast_images/build_fullframe_pairs.py \
  --trial both --horizon "$HORIZON" --size "$SIZE" --export-images

echo ""
echo "=== [2/2] Train Pix2Pix per trial ==="
for trial in trial1 trial2; do
  echo "----- train $trial -----"
  $PY -u ml/forecast_images/train_fullframe_pix2pix.py \
    --trial "$trial" --horizon "$HORIZON" --size "$SIZE" \
    --epochs "$EPOCHS" --batch-size 4 --lr 2e-4
done

echo ""
echo "=== Summaries ==="
$PY - <<PY
import json
from pathlib import Path
import pandas as pd
root = Path("ml/forecast_images/runs_fullframe")
rows = []
for p in sorted(root.glob("*/test_metrics.json")):
    d = json.loads(p.read_text())
    t = d["test"]; c = d["config"]
    rows.append({
        "run": p.parent.name,
        "trial": c["trial"],
        "n_train": c["n_train"],
        "n_test": c["n_test"],
        "test_l1": round(t["l1"], 4),
        "test_ssim": round(t["ssim"], 4),
        "veg_mae_fake": round(t["veg_mae_fake"], 4),
        "veg_mae_persist": round(t["veg_mae_persist"], 4),
        "beats_persist_veg": d["beats_persist_veg"],
    })
df = pd.DataFrame(rows)
df.to_csv(root / "summary_fullframe.csv", index=False)
print(df.to_string(index=False))
PY

echo "DONE $(date -Is)"

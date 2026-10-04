#!/usr/bin/env bash
# Aim 2: per-cup multimodal Pix2Pix (RGB crop + FastSAM seg), sensors/actuators required
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-/data/home/sai/anaconda3/bin/python3}"
HORIZON="${HORIZON:-5}"
SIZE="${SIZE:-128}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-8}"
MODES="${MODES:-rgb,seg}"
OUT="ml/forecast_images"
LOG="$OUT/cup_mm_pipeline_h${HORIZON}_s${SIZE}.log"
mkdir -p "$OUT"
exec > >(tee "$LOG") 2>&1

echo "=== Cup multimodal Aim2 h=${HORIZON} size=${SIZE} modes=${MODES} === $(date -Is)"

echo ""
echo "=== [1/2] Build dense cup pairs (rgb + seg) + sensors/actuators ==="
$PY -u ml/forecast_images/build_cup_multimodal_pairs.py \
  --trial both --horizon "$HORIZON" --size "$SIZE" \
  --modes "$MODES" --export-images

echo ""
echo "=== [2/2] Train Pix2Pix per trial × mode (climate required) ==="
IFS=',' read -r -a MODE_ARR <<< "$MODES"
for trial in trial1 trial2; do
  for mode in "${MODE_ARR[@]}"; do
    mode="$(echo "$mode" | xargs)"
    echo "----- train $trial mode=$mode -----"
    $PY -u ml/forecast_images/train_cup_multimodal_pix2pix.py \
      --trial "$trial" --mode "$mode" --horizon "$HORIZON" --size "$SIZE" \
      --epochs "$EPOCHS" --batch-size "$BATCH" --lr 2e-4
  done
done

echo ""
echo "=== Summaries ==="
$PY - <<'PY'
import json
from pathlib import Path
import pandas as pd
root = Path("ml/forecast_images/runs_cup_mm")
rows = []
for p in sorted(root.glob("*/test_metrics.json")):
    d = json.loads(p.read_text())
    t = d["test"]; c = d["config"]
    rows.append({
        "run": p.parent.name,
        "trial": c["trial"],
        "mode": c["mode"],
        "n_train": c["n_train"],
        "n_test": c["n_test"],
        "n_climate": c["n_climate"],
        "test_l1": round(t["l1"], 4),
        "test_ssim": round(t["ssim"], 4),
        "veg_mae_fake": round(t["veg_mae_fake"], 4),
        "veg_mae_persist": round(t["veg_mae_persist"], 4),
        "beats_persist_veg": d["beats_persist_veg"],
    })
df = pd.DataFrame(rows)
out = root / "summary_cup_mm.csv"
if len(df):
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
else:
    print("No test_metrics.json yet")
print(f"summary → {out}")
PY

echo "DONE $(date -Is)"

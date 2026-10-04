#!/usr/bin/env bash
# Aim 2 Phase B v2: train U-Net + TinyTransformer future-mask models on Trial1 & Trial2
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-/data/home/sai/anaconda3/bin/python3}"
OUT="ml/forecast_images/aim2_phase_b_mask"
LOG="$OUT/train_all.log"
mkdir -p "$OUT"
exec > >(tee "$LOG") 2>&1

echo "=== Aim 2 Phase B mask models === $(date -Is)"
EPOCHS="${EPOCHS:-80}"
for trial in trial1 trial2; do
  for model in unet tinyt; do
    echo ""
    echo "===== $trial / $model ====="
    $PY -u ml/forecast_images/train_aim2_phase_b_mask.py \
      --trial "$trial" --model "$model" --epochs "$EPOCHS" --batch-size 8 --lr 3e-4
  done
done

echo ""
echo "=== Aggregate test metrics ==="
$PY - <<'PY'
import json
from pathlib import Path
import pandas as pd
root = Path("ml/forecast_images/aim2_phase_b_mask")
rows = []
for p in sorted(root.glob("*/test_metrics.json")):
    d = json.loads(p.read_text())
    cfg = d["config"]
    t = d["test"]
    rows.append({
        "run": p.parent.name,
        "trial": cfg["trial"],
        "model": cfg["model"],
        "test_dice": round(t["dice"], 4),
        "persist_dice": round(t["dice_persist"], 4),
        "test_veg_mae": round(t["veg_mae"], 4),
        "persist_veg_mae": round(t["veg_mae_persist"], 4),
        "beats_persist_dice": d["beats_persist_dice"],
        "beats_persist_veg_mae": d["beats_persist_veg_mae"],
    })
df = pd.DataFrame(rows)
df.to_csv(root / "summary_all.csv", index=False)
print(df.to_string(index=False))
print("Wrote", root / "summary_all.csv")
PY

echo "DONE $(date -Is)"

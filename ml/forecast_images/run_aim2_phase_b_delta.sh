#!/usr/bin/env bash
# Aim 2 Phase B v3: Δ-mask residual (FOV-normalized soft ExGR growth)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-/data/home/sai/anaconda3/bin/python3}"
OUT="ml/forecast_images/aim2_phase_b_delta"
LOG="$OUT/train_all.log"
mkdir -p "$OUT"
exec > >(tee "$LOG") 2>&1

echo "=== Aim 2 Phase B Δ-mask residual === $(date -Is)"
EPOCHS="${EPOCHS:-100}"
for trial in trial1 trial2; do
  for model in unet tinyt; do
    echo ""
    echo "===== $trial / $model / delta ====="
    $PY -u ml/forecast_images/train_aim2_phase_b_delta_mask.py \
      --trial "$trial" --model "$model" --epochs "$EPOCHS" --batch-size 8 --lr 3e-4
  done
done

echo ""
echo "=== Aggregate ==="
$PY - <<'PY'
import json
from pathlib import Path
import pandas as pd
root = Path("ml/forecast_images/aim2_phase_b_delta")
rows = []
for p in sorted(root.glob("*/test_metrics.json")):
    d = json.loads(p.read_text())
    t = d["test"]
    c = d["config"]
    rows.append({
        "run": p.parent.name,
        "trial": c["trial"],
        "model": c["model"],
        "test_dice": round(t["dice"], 4),
        "persist_dice": round(t["dice_persist"], 4),
        "test_veg_mae": round(t["veg_mae"], 4),
        "persist_veg_mae": round(t["veg_mae_persist"], 4),
        "test_soft_l1": round(t["soft_l1"], 4),
        "persist_soft_l1": round(t["soft_l1_persist"], 4),
        "beats_dice": d["beats_persist_dice"],
        "beats_veg": d["beats_persist_veg_mae"],
        "beats_soft_l1": d["beats_persist_soft_l1"],
    })
df = pd.DataFrame(rows)
df.to_csv(root / "summary_all.csv", index=False)
print(df.to_string(index=False))
PY
echo "DONE $(date -Is)"

#!/usr/bin/env bash
# Aim 2 Phase B: phenotype consistency on existing Pix2Pix checkpoints
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-/data/home/sai/anaconda3/bin/python3}"
OUT="ml/forecast_images/aim2_phase_b"
mkdir -p "$OUT"
exec > >(tee "$OUT/phase_b.log") 2>&1
echo "=== Aim 2 Phase B === $(date -Is)"
$PY -u ml/forecast_images/eval_aim2_phase_b_phenotype.py --trials trial1 trial2
echo "DONE $(date -Is)"

#!/usr/bin/env python3
"""Aggregate per-seed Aim 1 plant-FW ablation CSVs into mean ± SD tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PLANT = Path(__file__).resolve().parent / "plant"


def aggregate(seeds: list[int], out_dir: Path = PLANT) -> pd.DataFrame:
    frames = []
    for s in seeds:
        p = out_dir / f"ablation_aim1_plant_fw_seed{s}.csv"
        if not p.exists():
            raise SystemExit(f"Missing per-seed table: {p}")
        df = pd.read_csv(p)
        df["seed"] = s
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    keys = ["feature_set", "model", "split"]
    metrics = ["mae_g", "rmse_g", "r2", "day_median_mae_g"]
    rows = []
    for key, g in all_df.groupby(keys, sort=True):
        rec = dict(zip(keys, key))
        rec["n_seeds"] = int(len(g))
        rec["n_features"] = int(g["n_features"].iloc[0])
        rec["n_rows"] = int(g["n_rows"].iloc[0])
        for m in metrics:
            vals = g[m].astype(float).to_numpy()
            rec[f"{m}_mean"] = float(vals.mean())
            rec[f"{m}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            rec[f"{m}_mean_pm_sd"] = f"{vals.mean():.2f} ± {vals.std(ddof=1):.2f}" if len(vals) > 1 else f"{vals.mean():.2f}"
        rows.append(rec)

    out = pd.DataFrame(rows)
    out_path = out_dir / "ablation_aim1_plant_fw_mean_sd.csv"
    out.to_csv(out_path, index=False)

    # Best model per feature_set × split (by mean MAE, exclude MeanBaseline)
    usable = out[out["model"] != "MeanBaseline"].copy()
    best = (
        usable.sort_values("mae_g_mean")
        .groupby(["feature_set", "split"], as_index=False)
        .first()
    )
    best_path = out_dir / "ablation_aim1_plant_fw_mean_sd_best.csv"
    best.to_csv(best_path, index=False)

    print(f"Wrote {out_path}")
    print(f"Wrote {best_path}")
    print("\n=== Headline LODO (mean ± SD MAE, g) — best model per feature set ===")
    lodo = best[best["split"] == "lodo"].sort_values("mae_g_mean")
    for _, r in lodo.iterrows():
        print(
            f"  {r['feature_set']:8s}  {r['model']:12s}  "
            f"MAE={r['mae_g_mean_pm_sd']}  R²={r['r2_mean']:.3f}±{r['r2_sd']:.3f}"
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Seeds whose per-seed CSVs already exist under ml/plant/",
    )
    args = p.parse_args()
    aggregate(args.seeds)


if __name__ == "__main__":
    main()

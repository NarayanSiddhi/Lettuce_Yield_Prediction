"""
Leave-one-out Ridge baseline for checkpoint fresh-weight prediction.

Usage:
  python3 train_checkpoint_baseline.py
  python3 train_checkpoint_baseline.py --trial trial2
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
META_COLS = {"trial", "day", "date", "target_median_fw_g", "window_start", "window_end", "window_days_count"}
TARGET = "target_median_fw_g"


def read_checkpoint_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {path}")
    return rows, list(rows[0].keys())


def feature_columns(all_cols: list[str], prefix: str | None, suffix: str = "_wmean") -> list[str]:
    out = []
    for c in all_cols:
        if c in META_COLS:
            continue
        if prefix and not c.startswith(prefix):
            continue
        if suffix and not c.endswith(suffix):
            continue
        out.append(c)
    return sorted(out)


def rows_to_xy(rows: list[dict], feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    y = np.array([float(r[TARGET]) for r in rows], dtype=float)
    x = np.empty((len(rows), len(feat_cols)), dtype=float)
    for i, r in enumerate(rows):
        for j, c in enumerate(feat_cols):
            v = r.get(c, "")
            x[i, j] = float(v) if v not in ("", None) else np.nan
    return x, y


def loo_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 10.0) -> tuple[np.ndarray, float]:
    """Leave-one-out predictions with Ridge + impute + scale."""
    n = len(y)
    preds = np.empty(n, dtype=float)
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        pipe.fit(x[mask], y[mask])
        preds[i] = float(pipe.predict(x[i : i + 1])[0])
    return preds, alpha


def eval_preds(y: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    return {
        "mae_g": float(mean_absolute_error(y, preds)),
        "rmse_g": float(math.sqrt(np.mean((y - preds) ** 2))),
        "r2": float(r2_score(y, preds)) if len(y) > 1 else float("nan"),
    }


def run_trial(trial: str, alpha: float) -> None:
    path = PROJECT_ROOT / f"master_checkpoint_dataset_{trial}.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: python3 build_master_dataset_from_cups.py")

    rows, all_cols = read_checkpoint_csv(path)
    rows = sorted(rows, key=lambda r: int(r["day"]))
    y = np.array([float(r[TARGET]) for r in rows])

    mean_baseline = np.full_like(y, y.mean())
    mean_metrics = eval_preds(y, mean_baseline)

    feature_sets = {
        "image": feature_columns(all_cols, "img_"),
        "sensors": feature_columns(all_cols, "env_") + feature_columns(all_cols, "nutrient_"),
        "combined": feature_columns(all_cols, None),
    }
    # Always include window size (data availability signal)
    extra = ["window_days_count"]
    for name in feature_sets:
        feature_sets[name] = sorted(set(feature_sets[name] + extra))

    print(f"\n=== {trial} ({len(rows)} checkpoints) ===")
    print(f"  target range: {y.min():.1f} – {y.max():.1f} g")
    print(f"  mean baseline  MAE={mean_metrics['mae_g']:.2f} g  R²={mean_metrics['r2']:.3f}")

    out_rows: list[dict] = []
    for name, cols in feature_sets.items():
        x, _ = rows_to_xy(rows, cols)
        preds, _ = loo_ridge(x, y, alpha=alpha)
        m = eval_preds(y, preds)
        print(
            f"  LOO Ridge [{name:8s}]  n_feat={len(cols):3d}  "
            f"MAE={m['mae_g']:.2f} g  RMSE={m['rmse_g']:.2f} g  R²={m['r2']:.3f}"
        )
        for r, pred in zip(rows, preds):
            out_rows.append(
                {
                    "trial": trial,
                    "model": name,
                    "day": r["day"],
                    "date": r["date"],
                    "target_median_fw_g": r[TARGET],
                    "pred_median_fw_g": f"{pred:.4f}",
                    "abs_error_g": f"{abs(pred - float(r[TARGET])):.4f}",
                    "window_days_count": r["window_days_count"],
                }
            )

    out_path = PROJECT_ROOT / f"baseline_checkpoint_predictions_{trial}.csv"
    fields = [
        "trial",
        "model",
        "day",
        "date",
        "target_median_fw_g",
        "pred_median_fw_g",
        "abs_error_g",
        "window_days_count",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"  saved: {out_path.name}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trial", default="trial1", choices=["trial1", "trial2", "both"])
    p.add_argument("--alpha", type=float, default=10.0, help="Ridge L2 strength (default 10)")
    args = p.parse_args()

    trials = ["trial1", "trial2"] if args.trial == "both" else [args.trial]
    for trial in trials:
        run_trial(trial, alpha=args.alpha)

    print("\nNote: only 8 labeled checkpoints per trial — treat metrics as exploratory, not final validation.")


if __name__ == "__main__":
    main()

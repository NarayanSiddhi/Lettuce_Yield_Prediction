"""
Compare crop-only, image-on-raw-weight, and hybrid (crop + residual ML) models.

Tray level (8 rows): ml/checkpoint_residual_trial{N}.csv
Plant level (79 rows): ml/plant/plant_residual_trial1.csv

Usage:
  python3 ml/train_residual_models.py --variant image
  python3 ml/train_residual_models.py --variant compact --trial trial1
  python3 ml/train_residual_models.py --variant all --trial both
  python3 ml/train_residual_models.py --plant-level --variant all
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent
PLANT_DIR = ML_DIR / "plant"
PROJECT_ROOT = ML_DIR.parent

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from train_yield_models import build_models, metrics

OBSERVED = "observed_g"
MODELLED = "modelled_g"
RESIDUAL = "target_residual_g"
PLANT_TARGET = "target_fw_g"

TRAY_META = {
    "sample_id",
    "trial",
    "day",
    "date",
    "window_start",
    "window_end",
    "label_source",
    "window_days_count",
    OBSERVED,
    MODELLED,
    "residual_g",
    "residual_pct",
    "crop_target_mid_g",
    "target_median_fw_g",
    RESIDUAL,
}

PLANT_META = TRAY_META | {
    PLANT_TARGET,
    "tray_median_fw_g",
    "tray_residual_g",
    "plant_id",
    "has_plant_cup_image",
    "n_plant_cup_days",
}


def select_feature_cols(df: pd.DataFrame, variant: str, *, plant: bool) -> list[str]:
    meta = PLANT_META if plant else TRAY_META
    cols: list[str] = []
    for c in df.columns:
        if c in meta:
            continue
        if c.endswith(("_wmin", "_wmax")):
            continue
        if not (pd.api.types.is_numeric_dtype(df[c]) or pd.to_numeric(df[c], errors="coerce").notna().any()):
            continue
        if variant == "image":
            if c.startswith("img_") or c.startswith("win_img_") or c.startswith("plant_img_"):
                cols.append(c)
        elif variant == "compact":
            if c.endswith("_wmean") or c.startswith("win_img_") or c.startswith("plant_img_"):
                cols.append(c)
        elif variant == "sensors":
            if c.startswith("env_") or c.startswith("nutrient_"):
                cols.append(c)
        elif variant == "full":
            cols.append(c)
        else:
            raise ValueError(variant)
    extra = ("day", "window_days_count")
    if plant:
        extra = ("day", "plant_id", "window_days_count", "has_plant_cup_image", "n_plant_cup_days")
    for c in extra:
        if c in df.columns and c not in cols:
            cols.append(c)
    return sorted({c for c in cols if df[c].notna().any()})


def lodo_predict_rows(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(y)
    pred = np.empty(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        model.fit(X[mask], y[mask])
        pred[i] = float(model.predict(X[i : i + 1])[0])
    return pred


def lodo_predict_days(model, X: np.ndarray, y: np.ndarray, days: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    for d in np.unique(days):
        te = days == d
        tr = ~te
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te]).astype(float)
    return pred


@dataclass(frozen=True)
class SetupResult:
    setup: str
    model: str
    mae_g: float
    rmse_g: float
    r2: float
    n_features: int
    n_rows: int
    split: str
    variant: str = ""


def load_tray_table(trial: str) -> pd.DataFrame:
    path = ML_DIR / f"checkpoint_residual_{trial}.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: python3 build_residual_checkpoint_dataset.py")
    return pd.read_csv(path)


def load_plant_table() -> pd.DataFrame:
    path = PLANT_DIR / "plant_residual_trial1.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: python3 build_residual_checkpoint_dataset.py")
    return pd.read_csv(path)


def matrix(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    return df[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def day_median_mae(days: np.ndarray, y: np.ndarray, pred: np.ndarray) -> float:
    errs = []
    for d in np.unique(days):
        mask = days == d
        errs.append(abs(float(np.median(pred[mask])) - float(np.median(y[mask]))))
    return float(np.mean(errs)) if errs else float("nan")


def run_comparison(
    df: pd.DataFrame,
    variant: str,
    seed: int,
    *,
    plant: bool,
    split: str,
) -> tuple[list[SetupResult], list[str]]:
    target_col = PLANT_TARGET if plant else OBSERVED
    observed = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    modelled = pd.to_numeric(df[MODELLED], errors="coerce").to_numpy(dtype=float)
    days = pd.to_numeric(df["day"], errors="coerce").to_numpy(dtype=int)
    feat_cols = select_feature_cols(df, variant, plant=plant)
    X = matrix(df, feat_cols)

    if split == "lodo" and plant:
        predict = lambda m, yt: lodo_predict_days(m, X, yt, days)
    else:
        predict = lambda m, yt: lodo_predict_rows(m, X, yt)

    results: list[SetupResult] = []
    n = len(df)

    mae, rmse, r2 = metrics(observed, modelled)
    results.append(
        SetupResult("crop_only", "physics", mae, rmse, r2, 0, n, split, variant)
    )

    mean_pred = np.full_like(observed, float(np.mean(observed)))
    mae, rmse, r2 = metrics(observed, mean_pred)
    results.append(
        SetupResult("mean_baseline", "mean", mae, rmse, r2, 0, n, split, variant)
    )

    residual = observed - modelled
    for name, model in build_models(random_state=seed):
        pred_raw = predict(model, observed)
        mae, rmse, r2 = metrics(observed, pred_raw)
        results.append(
            SetupResult(f"{variant}_raw", name, mae, rmse, r2, len(feat_cols), n, split, variant)
        )

        pred_res = predict(model, residual)
        hybrid = modelled + pred_res
        mae, rmse, r2 = metrics(observed, hybrid)
        results.append(
            SetupResult(f"{variant}_hybrid", name, mae, rmse, r2, len(feat_cols), n, split, variant)
        )

    return results, feat_cols


def print_results(title: str, results: list[SetupResult]) -> None:
    print(f"\n=== {title} ===")
    print(f"{'setup':20s} {'model':14s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s} {'n_feat':>6s}")
    for r in sorted(results, key=lambda x: x.mae_g):
        print(
            f"{r.setup:20s} {r.model:14s} {r.mae_g:8.2f} {r.rmse_g:8.2f} "
            f"{r.r2:8.3f} {r.n_features:6d}"
        )


def save_summary(path: Path, results: list[SetupResult], extra: dict | None = None) -> None:
    fields = ["level", "trial", "split", "variant", "setup", "model", "n_rows",
              "mae_g", "rmse_g", "r2", "n_features"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {
                "setup": r.setup,
                "model": r.model,
                "n_rows": r.n_rows,
                "mae_g": f"{r.mae_g:.4f}",
                "rmse_g": f"{r.rmse_g:.4f}",
                "r2": f"{r.r2:.6f}",
                "n_features": r.n_features,
                "split": r.split,
                "variant": r.variant,
            }
            if extra:
                row = {**extra, **row}
            w.writerow(row)


def run_tray(args) -> list[SetupResult]:
    variants = ["image", "compact", "sensors"] if args.variant == "all" else [args.variant]
    trials = ["trial1", "trial2"] if args.trial == "both" else [args.trial]
    all_results: list[SetupResult] = []

    for trial in trials:
        df = load_tray_table(trial)
        for variant in variants:
            results, feat_cols = run_comparison(
                df, variant, args.seed, plant=False, split="lodo"
            )
            print_results(
                f"Tray LODO {trial} / {variant} ({len(df)} rows, {len(feat_cols)} features)",
                results,
            )
            out = ML_DIR / f"results_residual_{trial}_{variant}_lodo.csv"
            save_summary(out, results, extra={"level": "tray", "trial": trial})
            print(f"Saved: {out.relative_to(PROJECT_ROOT)}")
            all_results.extend(results)

    if args.holdout_trial2:
        df1 = load_tray_table("trial1")
        df2 = load_tray_table("trial2")
        for variant in variants:
            feat_cols = select_feature_cols(df1, variant, plant=False)
            X1, X2 = matrix(df1, feat_cols), matrix(df2, feat_cols)
            y2 = pd.to_numeric(df2[OBSERVED], errors="coerce").to_numpy(dtype=float)
            m2 = pd.to_numeric(df2[MODELLED], errors="coerce").to_numpy(dtype=float)
            r1 = pd.to_numeric(df1[RESIDUAL], errors="coerce").to_numpy(dtype=float)
            holdout: list[SetupResult] = []
            mae, rmse, r2 = metrics(y2, m2)
            holdout.append(
                SetupResult("crop_only_holdout", "physics", mae, rmse, r2, 0, len(df2), "holdout", variant)
            )
            for name, model in build_models(random_state=args.seed):
                model.fit(X1, pd.to_numeric(df1[OBSERVED], errors="coerce").to_numpy(dtype=float))
                mae, rmse, r2 = metrics(y2, model.predict(X2))
                holdout.append(
                    SetupResult(f"{variant}_raw_holdout", name, mae, rmse, r2, len(feat_cols), len(df2), "holdout", variant)
                )
                model.fit(X1, r1)
                mae, rmse, r2 = metrics(y2, m2 + model.predict(X2))
                holdout.append(
                    SetupResult(f"{variant}_hybrid_holdout", name, mae, rmse, r2, len(feat_cols), len(df2), "holdout", variant)
                )
            print_results(f"Hold-out trial2 / {variant}", holdout)
            out = ML_DIR / f"results_residual_holdout_trial2_{variant}.csv"
            save_summary(out, holdout, extra={"level": "tray", "trial": "trial2"})
            all_results.extend(holdout)

    return all_results


def run_plant(args) -> list[SetupResult]:
    variants = ["image", "compact", "sensors"] if args.variant == "all" else [args.variant]
    df = load_plant_table()
    days = pd.to_numeric(df["day"], errors="coerce").to_numpy(dtype=int)
    y = pd.to_numeric(df[PLANT_TARGET], errors="coerce").to_numpy(dtype=float)
    all_results: list[SetupResult] = []

    for variant in variants:
        for split in ("lodo", "loo"):
            results, feat_cols = run_comparison(
                df, variant, args.seed, plant=True, split=split
            )
            print_results(
                f"Plant {split} / {variant} ({len(df)} rows, {len(feat_cols)} features)",
                results,
            )
            # day-median MAE for best hybrid
            feat_cols_use = feat_cols
            X = matrix(df, feat_cols_use)
            modelled = pd.to_numeric(df[MODELLED], errors="coerce").to_numpy(dtype=float)
            residual = y - modelled
            best_hybrid_mae = min(
                r.mae_g for r in results if r.setup.endswith("_hybrid") and r.model != "mean"
            )
            best = min(
                (r for r in results if r.setup.endswith("_hybrid")),
                key=lambda r: r.mae_g,
            )
            print(f"  Best hybrid: {best.model} plant MAE={best.mae_g:.2f} g")

            out = PLANT_DIR / f"results_residual_plant_{variant}_{split}.csv"
            save_summary(out, results, extra={"level": "plant", "trial": "trial1"})
            print(f"Saved: {out.relative_to(PROJECT_ROOT)}")
            all_results.extend(results)

    combined = PLANT_DIR / "results_residual_plant_all.csv"
    save_summary(combined, all_results, extra={"level": "plant", "trial": "trial1"})
    return all_results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trial", default="trial1", choices=["trial1", "trial2", "both"])
    p.add_argument("--variant", default="image", choices=["image", "compact", "sensors", "all"])
    p.add_argument("--plant-level", action="store_true")
    p.add_argument("--holdout-trial2", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.plant_level:
        run_plant(args)
    else:
        run_tray(args)

    print(
        "\nCompare crop_only vs *_hybrid MAE on observed weight. "
        "Plant LODO is the fair split; plant LOO is optimistic."
    )


if __name__ == "__main__":
    main()

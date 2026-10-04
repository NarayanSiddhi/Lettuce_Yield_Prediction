#!/usr/bin/env python3
"""
Aim 2 hard generalization: leave-one-trial-out tray t→t+4 FW.

Train on Trial 1 → test Trial 2 (and reverse).
Routes: Persist, GrowMean, Direct FW, Phenotype bridge (canopy→FW).

N=2 cycles only — report as hard test, not definitive.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
for k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(k, "1")

ML_DIR = Path(__file__).resolve().parent
OUT_DIR = ML_DIR / "aim2_loto"
HORIZON = 4

SENSOR_COLS = [
    "env_day_air_temp_c_mean_wmean",
    "env_night_air_temp_c_mean_wmean",
    "env_day_humidity_pct_mean_wmean",
    "env_night_vpd_pa_mean_wmean",
    "env_day_vpd_pa_mean_wmean",
    "env_day_co2_ppm_mean_wmean",
    "env_act_day_light_duration_s_sum_wmean",
    "nutrient_ec_us_cm_mean_wmean",
    "exp_day_wmean",
]
IMAGE_COLS = [
    "img_mask_area_sum_wmean",
    "img_mask_area_mean_wmean",
    "img_coverage_frac_wmean",
    "img_plant_color_mean_wmean",
    "win_img_mask_area_sum_slope",
    "win_img_mask_area_sum_delta",
    "win_img_coverage_frac_slope",
]
PHENO = "img_mask_area_sum_wmean"


@dataclass
class Row:
    seed: int
    fold: str  # t1_to_t2 | t2_to_t1
    route: str
    model: str
    n_train: int
    n_test: int
    mae_g: float
    rmse_g: float
    r2: float
    mae_vs_persist: float
    canopy_mae_px: float


def metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mae = float(np.mean(np.abs(y - pred)))
    rmse = float(math.sqrt(float(np.mean((y - pred) ** 2))))
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else float("inf"))
    return mae, rmse, r2


def zoo(seed: int):
    models = [
        (
            "Ridge",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", Ridge(alpha=10.0)),
                ]
            ),
        ),
        (
            "ElasticNet",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        ElasticNet(
                            alpha=0.05,
                            l1_ratio=0.25,
                            max_iter=50_000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
        (
            "GradBoost",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "model",
                        GradientBoostingRegressor(
                            n_estimators=100, max_depth=2, random_state=seed
                        ),
                    ),
                ]
            ),
        ),
        (
            "RandomForest",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=200, max_depth=3, random_state=seed, n_jobs=1
                        ),
                    ),
                ]
            ),
        ),
    ]
    try:
        from xgboost import XGBRegressor

        models.append(
            (
                "XGBoost",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        (
                            "model",
                            XGBRegressor(
                                n_estimators=80,
                                max_depth=2,
                                learning_rate=0.1,
                                subsample=0.9,
                                colsample_bytree=0.8,
                                reg_lambda=2.0,
                                random_state=seed,
                                n_jobs=1,
                            ),
                        ),
                    ]
                ),
            )
        )
    except Exception:
        pass
    return models


def build_tray_pairs() -> pd.DataFrame:
    rows = []
    for trial in ("trial1", "trial2"):
        df = pd.read_csv(ML_DIR / f"checkpoint_residual_{trial}.csv")
        by_day = {int(r["day"]): r for _, r in df.iterrows()}
        for day_t, row_t in by_day.items():
            day_f = day_t + HORIZON
            if day_f not in by_day:
                continue
            row_f = by_day[day_f]
            prev = day_t - HORIZON
            gain = (
                float(row_t["observed_g"]) - float(by_day[prev]["observed_g"])
                if prev in by_day
                else float("nan")
            )
            feat = {
                "trial": trial,
                "day_t": day_t,
                "day_future": day_f,
                "fw_t": float(row_t["observed_g"]),
                "fw_future": float(row_f["observed_g"]),
                "gain_prev_4d": gain,
                "pheno_mask_t": float(row_t[PHENO]) if PHENO in row_t and pd.notna(row_t[PHENO]) else float("nan"),
                "pheno_mask_future": float(row_f[PHENO])
                if PHENO in row_f and pd.notna(row_f[PHENO])
                else float("nan"),
            }
            for c in SENSOR_COLS + IMAGE_COLS:
                if c in row_t.index:
                    feat[c] = float(row_t[c]) if pd.notna(row_t[c]) else np.nan
            rows.append(feat)
    return pd.DataFrame(rows)


def mat(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    present = [c for c in cols if c in df.columns]
    return df[present].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), present


def best_model_predict(Xtr, ytr, Xte, seed: int) -> tuple[np.ndarray, str]:
    """Pick model by leave-one-row MAE on train (tiny n). Drops NaN labels."""
    ytr = np.asarray(ytr, dtype=float)
    ok = np.isfinite(ytr) & np.all(np.isfinite(Xtr) | True, axis=1)
    # X may still have NaNs — pipelines impute features; only require finite y
    ok = np.isfinite(ytr)
    if ok.sum() < 2:
        # fallback: mean predictor
        fill = float(np.nanmean(ytr)) if np.isfinite(ytr).any() else 0.0
        return np.full(len(Xte), fill, dtype=float), "MeanFill"
    Xtr, ytr = Xtr[ok], ytr[ok]

    best_name, best_mae, best_pred = "ElasticNet", float("inf"), None
    n = len(ytr)
    for name, model in zoo(seed):
        if n < 3:
            m = clone(model)
            m.fit(Xtr, ytr)
            return m.predict(Xte).astype(float), name
        loo = np.empty(n, dtype=float)
        for i in range(n):
            m = clone(model)
            idx = np.ones(n, dtype=bool)
            idx[i] = False
            m.fit(Xtr[idx], ytr[idx])
            loo[i] = float(m.predict(Xtr[i : i + 1])[0])
        mae = float(np.mean(np.abs(ytr - loo)))
        if mae < best_mae:
            best_mae, best_name = mae, name
            m = clone(dict(zoo(seed))[name])
            m.fit(Xtr, ytr)
            best_pred = m.predict(Xte).astype(float)
    assert best_pred is not None
    return best_pred, best_name


def run_fold(pairs: pd.DataFrame, train_trial: str, test_trial: str, seed: int) -> tuple[list[Row], pd.DataFrame]:
    tr = pairs[pairs.trial == train_trial].copy()
    te = pairs[pairs.trial == test_trial].copy()
    fold = "t1_to_t2" if train_trial == "trial1" else "t2_to_t1"

    ytr = tr["fw_future"].to_numpy(dtype=float)
    yte = te["fw_future"].to_numpy(dtype=float)
    fw_t_te = te["fw_t"].to_numpy(dtype=float)
    fw_t_tr = tr["fw_t"].to_numpy(dtype=float)

    persist = fw_t_te.copy()
    persist_mae = float(np.mean(np.abs(yte - persist)))
    mean_gain = float(np.mean(ytr - fw_t_tr))
    grow = fw_t_te + mean_gain

    direct_cols = [c for c in SENSOR_COLS + ["fw_t", "gain_prev_4d"] if c in tr.columns]
    Xtr_d, direct_cols = mat(tr, direct_cols)
    Xte_d, _ = mat(te, direct_cols)
    pred_direct, direct_name = best_model_predict(Xtr_d, ytr, Xte_d, seed)

    # Canopy
    canopy_cols = [c for c in SENSOR_COLS + IMAGE_COLS + ["fw_t", "gain_prev_4d", "pheno_mask_t"] if c in tr.columns]
    Xtr_c, canopy_cols = mat(tr, canopy_cols)
    Xte_c, _ = mat(te, canopy_cols)
    ytr_m = tr["pheno_mask_future"].to_numpy(dtype=float)
    yte_m = te["pheno_mask_future"].to_numpy(dtype=float)
    pred_mask, canopy_name = best_model_predict(Xtr_c, ytr_m, Xte_c, seed)
    # Canopy MAE only where test phenotype is observed
    te_ok = np.isfinite(yte_m) & np.isfinite(pred_mask)
    canopy_mae = float(np.mean(np.abs(yte_m[te_ok] - pred_mask[te_ok]))) if te_ok.any() else float("nan")
    pheno_t = te["pheno_mask_t"].to_numpy(dtype=float)
    cp_ok = np.isfinite(yte_m) & np.isfinite(pheno_t)
    canopy_persist = float(np.mean(np.abs(yte_m[cp_ok] - pheno_t[cp_ok]))) if cp_ok.any() else float("nan")
    # If predicted mask is nan, fall back to pheno_t / train median
    pred_mask = np.where(
        np.isfinite(pred_mask),
        pred_mask,
        np.where(np.isfinite(pheno_t), pheno_t, float(np.nanmedian(ytr_m))),
    )

    # Bridge: train on true future mask, test on predicted
    bridge_cols = [c for c in SENSOR_COLS + ["fw_t", "gain_prev_4d"] if c in tr.columns]
    Xtr_b = tr[bridge_cols].copy()
    Xtr_b["pheno_in"] = ytr_m
    Xtr_b["day_future_feat"] = tr["day_future"].astype(float)
    Xte_b = te[bridge_cols].copy()
    Xte_b["pheno_in"] = pred_mask
    Xte_b["day_future_feat"] = te["day_future"].astype(float)
    Xtr_b_m = Xtr_b.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    Xte_b_m = Xte_b.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    pred_pheno, bridge_name = best_model_predict(Xtr_b_m, ytr, Xte_b_m, seed)

    print(
        f"  [{fold} seed={seed}] persist={persist_mae:.2f}  "
        f"direct={direct_name}:{metrics(yte, pred_direct)[0]:.2f}  "
        f"bridge={bridge_name}←{canopy_name}:{metrics(yte, pred_pheno)[0]:.2f}  "
        f"canopy_px={canopy_mae:.0f} (persist {canopy_persist:.0f})"
    )

    rows = []
    for route, model, pred, cmae in [
        ("persist", "Persist", persist, canopy_persist),
        ("grow_mean", "GrowMean", grow, float("nan")),
        ("direct_fw", direct_name, pred_direct, float("nan")),
        ("phenotype_bridge", f"{bridge_name}←{canopy_name}", pred_pheno, canopy_mae),
        ("canopy_only", canopy_name, None, canopy_mae),
    ]:
        if pred is None:
            mae = rmse = r2 = float("nan")
            vs = float("nan")
        else:
            mae, rmse, r2 = metrics(yte, pred)
            vs = persist_mae - mae
        rows.append(
            Row(seed, fold, route, model, len(tr), len(te), mae, rmse, r2, vs, cmae)
        )

    pred_df = te[["trial", "day_t", "day_future", "fw_t", "fw_future", "pheno_mask_t", "pheno_mask_future"]].copy()
    pred_df["fold"] = fold
    pred_df["seed"] = seed
    pred_df["pred_fw_persist"] = persist
    pred_df["pred_fw_growmean"] = grow
    pred_df["pred_fw_direct"] = pred_direct
    pred_df["pred_fw_phenotype"] = pred_pheno
    pred_df["pred_mask_future"] = pred_mask
    return rows, pred_df


def aggregate(results: list[Row]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in results])
    rows = []
    for key, g in df.groupby(["fold", "route"], sort=True):
        # pick majority / first model label from mode
        model = g["model"].mode().iloc[0] if len(g) else ""
        rec = {"fold": key[0], "route": key[1], "model": model, "n_seeds": len(g)}
        for col in ("mae_g", "rmse_g", "r2", "mae_vs_persist", "canopy_mae_px"):
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                rec[f"{col}_mean_pm_sd"] = "—"
                rec[f"{col}_mean"] = float("nan")
            elif len(vals) == 1:
                rec[f"{col}_mean_pm_sd"] = f"{vals[0]:.2f}"
                rec[f"{col}_mean"] = float(vals[0])
            else:
                rec[f"{col}_mean_pm_sd"] = f"{vals.mean():.2f} ± {vals.std(ddof=1):.2f}"
                rec[f"{col}_mean"] = float(vals.mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Aim 2 LOTO tray t→t+4 (T1↔T2)")
    pairs = build_tray_pairs()
    pairs.to_csv(OUT_DIR / "tray_pairs_aim2_loto.csv", index=False)
    print(f"  pairs={len(pairs)}  T1={sum(pairs.trial=='trial1')}  T2={sum(pairs.trial=='trial2')}")

    all_rows: list[Row] = []
    preds = []
    for seed in args.seeds:
        print(f"\n=== seed={seed} ===")
        for tr_trial, te_trial in (("trial1", "trial2"), ("trial2", "trial1")):
            rows, pred = run_fold(pairs, tr_trial, te_trial, seed)
            all_rows.extend(rows)
            preds.append(pred)

    pd.DataFrame([asdict(r) for r in all_rows]).to_csv(OUT_DIR / "results_aim2_loto_per_seed.csv", index=False)
    agg = aggregate(all_rows)
    agg.to_csv(OUT_DIR / "results_aim2_loto_mean_sd.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(OUT_DIR / "predictions_aim2_loto.csv", index=False)

    summary = {"seeds": args.seeds, "n_pairs": int(len(pairs)), "folds": {}}
    print("\n=== HEADLINE mean ± SD ===")
    for fold in ("t1_to_t2", "t2_to_t1"):
        summary["folds"][fold] = {}
        print(f"\n-- {fold} --")
        sub = agg[agg.fold == fold]
        for route in ("persist", "grow_mean", "direct_fw", "phenotype_bridge", "canopy_only"):
            r = sub[sub.route == route]
            if r.empty:
                continue
            row = r.iloc[0]
            print(
                f"  {route:18s}  {str(row['model']):28s}  "
                f"FW={row['mae_g_mean_pm_sd']:14s}  canopy={row['canopy_mae_px_mean_pm_sd']}"
            )
            summary["folds"][fold][route] = {
                "model": row["model"],
                "mae_g": row["mae_g_mean_pm_sd"],
                "canopy_px": row["canopy_mae_px_mean_pm_sd"],
            }
    (OUT_DIR / "aim2_loto_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR}/")


if __name__ == "__main__":
    main()

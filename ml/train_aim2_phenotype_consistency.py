#!/usr/bin/env python3
"""
Aim 2 Phase A (tuned) — phenotype-consistency forecast.

Fixes vs first pass:
  - Match proven next-checkpoint HPs (ElasticNet α=0.05, light GradBoost/XGB)
  - Direct FW uses sensors + fw_t (+ gain), not the noisy full compact dump
  - Small nested LODO HP search per outer fold (--tune)
  - GrowMean baseline alongside Persist

Routes: persist | grow_mean | direct_fw | phenotype_bridge | canopy_only
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
PROJECT = ML_DIR.parent
PLANT_DIR = ML_DIR / "plant"
HORIZON = 4

PHENOTYPE_COLS_T = [
    "plant_img_mask_area_px_wmean",
    "plant_img_mask_area_px_last",
    "img_mask_area_sum_wmean",
    "img_mask_area_mean_wmean",
    "img_coverage_frac_wmean",
]
SENSOR_COLS = [
    "env_day_air_temp_c_mean_wmean",
    "env_night_air_temp_c_mean_wmean",
    "env_day_humidity_pct_mean_wmean",
    "env_night_humidity_pct_mean_wmean",
    "env_day_vpd_pa_mean_wmean",
    "env_night_vpd_pa_mean_wmean",
    "env_day_co2_ppm_mean_wmean",
    "env_act_day_light_duration_s_sum_wmean",
    "nutrient_ec_us_cm_mean_wmean",
    "exp_day_wmean",
]
DIRECT_EXTRA = ["fw_t", "gain_prev_4d"]


@dataclass
class RowResult:
    seed: int
    route: str
    model: str
    split: str
    n_rows: int
    n_features: int
    mae_g: float
    rmse_g: float
    r2: float
    mae_vs_persist: float
    canopy_mae_px: float
    consistency_mae_g: float


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 2:
        return float("nan")
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return 1.0 - (ss_res / ss_tot if ss_tot > 0 else float("inf"))


def metrics(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    mae = float(np.mean(np.abs(y - pred)))
    rmse = float(math.sqrt(float(np.mean((y - pred) ** 2))))
    return mae, rmse, _r2(y, pred)


def model_zoo(seed: int) -> list[tuple[str, object]]:
    """Proven next-checkpoint-style configs + light variants for tuning."""
    models: list[tuple[str, object]] = [
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
            "ElasticNet_wide",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        ElasticNet(
                            alpha=0.2,
                            l1_ratio=0.5,
                            max_iter=50_000,
                            random_state=seed,
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
                            n_estimators=200,
                            max_depth=3,
                            random_state=seed,
                            n_jobs=1,
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
                            n_estimators=100,
                            max_depth=2,
                            random_state=seed,
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
        models.append(
            (
                "XGBoost_deeper",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        (
                            "model",
                            XGBRegressor(
                                n_estimators=120,
                                max_depth=3,
                                learning_rate=0.08,
                                subsample=0.9,
                                colsample_bytree=0.8,
                                reg_lambda=3.0,
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


def cup_mask_lookup() -> pd.DataFrame:
    cup = pd.read_csv(PROJECT / "segmentation" / "daily_cup_features.csv")
    cup["date"] = pd.to_datetime(cup["date"]).dt.normalize()
    cup = (
        cup.sort_values(["trial", "cup_id", "date", "selection_score"], ascending=[True, True, True, False])
        .groupby(["trial", "cup_id", "date"], as_index=False)
        .first()
    )
    rows = []
    for trial, g in cup.groupby("trial"):
        drivers = pd.read_csv(PROJECT / "Lettuce_model" / "data" / f"{trial}_daily_drivers.csv")
        drivers["date"] = pd.to_datetime(drivers["date"]).dt.normalize()
        m = g.merge(drivers[["date", "DAT"]], on="date", how="inner")
        rows.append(m[["trial", "cup_id", "DAT", "mask_area_px", "plant_color_frac"]])
    return pd.concat(rows, ignore_index=True)


def build_pairs() -> pd.DataFrame:
    plant = pd.read_csv(PLANT_DIR / "plant_checkpoint_trial1_compact.csv")
    residual = pd.read_csv(PLANT_DIR / "plant_residual_trial1.csv")[
        ["plant_id", "day", "modelled_g", "target_residual_g"]
    ]
    plant = plant.merge(residual, on=["plant_id", "day"], how="left")
    masks = cup_mask_lookup()
    masks_t1 = masks[masks["trial"] == "trial1"].copy()

    def cup_mask(plant_id: int, dat: int) -> float:
        hit = masks_t1[(masks_t1["cup_id"] == plant_id) & (masks_t1["DAT"] == dat)]
        return float(hit["mask_area_px"].iloc[0]) if len(hit) else float("nan")

    def cup_color(plant_id: int, dat: int) -> float:
        hit = masks_t1[(masks_t1["cup_id"] == plant_id) & (masks_t1["DAT"] == dat)]
        return float(hit["plant_color_frac"].iloc[0]) if len(hit) else float("nan")

    def pheno_from_row(row: pd.Series, plant_id: int, dat: int) -> float:
        for v in (
            cup_mask(plant_id, dat),
            row.get("plant_img_mask_area_px_wmean"),
            row.get("plant_img_mask_area_px_last"),
            row.get("img_mask_area_mean_wmean"),
            row.get("img_mask_area_sum_wmean"),
        ):
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                return fv
        return float("nan")

    pull = [
        c
        for c in plant.columns
        if c
        in set(
            SENSOR_COLS
            + PHENOTYPE_COLS_T
            + [
                "has_plant_cup_image",
                "win_img_mask_area_sum_slope",
                "win_img_mask_area_sum_delta",
                "win_img_coverage_frac_slope",
            ]
        )
    ]

    rows = []
    for plant_id, g in plant.groupby("plant_id"):
        by_day = {int(r["day"]): r for _, r in g.sort_values("day").iterrows()}
        for day_t, row_t in by_day.items():
            day_f = day_t + HORIZON
            if day_f not in by_day:
                continue
            row_f = by_day[day_f]
            prev = day_t - HORIZON
            gain = (
                float(row_t["target_fw_g"]) - float(by_day[prev]["target_fw_g"])
                if prev in by_day
                else float("nan")
            )
            feat = {
                "trial": "trial1",
                "plant_id": int(plant_id),
                "day_t": day_t,
                "day_future": day_f,
                "fw_t": float(row_t["target_fw_g"]),
                "fw_future": float(row_f["target_fw_g"]),
                "gain_prev_4d": gain,
                "cup_mask_area_t": cup_mask(int(plant_id), day_t),
                "cup_color_t": cup_color(int(plant_id), day_t),
                "pheno_mask_t": pheno_from_row(row_t, int(plant_id), day_t),
                "pheno_mask_future": pheno_from_row(row_f, int(plant_id), day_f),
            }
            for c in pull:
                val = row_t[c]
                feat[c] = float(val) if pd.notna(val) else np.nan
            rows.append(feat)
    return pd.DataFrame(rows)


def feature_matrix(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    present = [c for c in cols if c in df.columns]
    return df[present].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), present


def clone_fit_predict(model, X_tr, y_tr, X_te) -> np.ndarray:
    from sklearn.base import clone

    m = clone(model)
    m.fit(X_tr, y_tr)
    return m.predict(X_te).astype(float)


def lodo_best_model(
    X: np.ndarray,
    y: np.ndarray,
    days: np.ndarray,
    seed: int,
    tune: bool,
) -> tuple[np.ndarray, str, float]:
    """Outer LODO; optionally pick best zoo model by inner-train MAE each fold."""
    zoo = model_zoo(seed)
    if not tune:
        # Fixed: try all models on full outer LODO, keep best overall name's preds
        best_name, best_pred, best_mae = "", None, float("inf")
        for name, model in zoo:
            pred = np.full(len(y), np.nan, dtype=float)
            for d in np.unique(days):
                te = days == d
                tr = ~te
                if tr.sum() < 5:
                    continue
                pred[te] = clone_fit_predict(model, X[tr], y[tr], X[te])
            mae = float(np.nanmean(np.abs(y - pred)))
            if mae < best_mae:
                best_mae, best_name, best_pred = mae, name, pred
        assert best_pred is not None
        return best_pred, best_name, best_mae

    # Tuned: per outer fold, choose model by inner LODO MAE on train days
    pred_out = np.full(len(y), np.nan, dtype=float)
    chosen = []
    for d in np.unique(days):
        te = days == d
        tr = ~te
        if tr.sum() < 8:
            # fallback ElasticNet
            model = dict(zoo)["ElasticNet"]
            pred_out[te] = clone_fit_predict(model, X[tr], y[tr], X[te])
            chosen.append("ElasticNet")
            continue
        Xtr, ytr, dtr = X[tr], y[tr], days[tr]
        best_inner_name, best_inner_mae = "ElasticNet", float("inf")
        for name, model in zoo:
            inner_pred = np.full(len(ytr), np.nan, dtype=float)
            for d2 in np.unique(dtr):
                te2 = dtr == d2
                tr2 = ~te2
                if tr2.sum() < 5:
                    continue
                inner_pred[te2] = clone_fit_predict(model, Xtr[tr2], ytr[tr2], Xtr[te2])
            mae = float(np.nanmean(np.abs(ytr - inner_pred)))
            if mae < best_inner_mae:
                best_inner_mae, best_inner_name = mae, name
        model = dict(zoo)[best_inner_name]
        pred_out[te] = clone_fit_predict(model, X[tr], y[tr], X[te])
        chosen.append(best_inner_name)
    # majority label for reporting
    from collections import Counter

    name = Counter(chosen).most_common(1)[0][0]
    mae = float(np.nanmean(np.abs(y - pred_out)))
    return pred_out, name, mae


def run_seed(pairs: pd.DataFrame, seed: int, tune: bool) -> tuple[list[RowResult], pd.DataFrame]:
    usable = pairs[np.isfinite(pairs["pheno_mask_future"].to_numpy(dtype=float))].copy()
    usable = usable[np.isfinite(usable["fw_future"].to_numpy(dtype=float))].copy()
    days = usable["day_future"].to_numpy(dtype=int)
    y_fw = usable["fw_future"].to_numpy(dtype=float)
    y_mask = usable["pheno_mask_future"].to_numpy(dtype=float)
    fw_t = usable["fw_t"].to_numpy(dtype=float)
    persist_mae = float(np.mean(np.abs(y_fw - fw_t)))

    # GrowMean baseline
    gain = y_fw - fw_t
    grow = np.empty_like(y_fw)
    for d in np.unique(days):
        te = days == d
        tr = ~te
        mean_gain = float(np.mean(gain[tr])) if tr.any() else float(np.mean(gain))
        grow[te] = fw_t[te] + mean_gain
    grow_mae = float(np.mean(np.abs(y_fw - grow)))

    # Direct: sensors + fw history (proven)
    direct_cols = [c for c in (SENSOR_COLS + DIRECT_EXTRA) if c in usable.columns]
    X_direct, direct_cols = feature_matrix(usable, direct_cols)

    # Canopy: sensors + phenotype@t + fw
    canopy_cols = [
        c
        for c in (SENSOR_COLS + DIRECT_EXTRA + ["pheno_mask_t", "cup_color_t"] + PHENOTYPE_COLS_T)
        if c in usable.columns
    ]
    X_canopy, canopy_cols = feature_matrix(usable, canopy_cols)

    print(
        f"\n=== seed={seed} tune={tune}  n={len(usable)}  "
        f"persist={persist_mae:.2f}g  growMean={grow_mae:.2f}g ==="
    )

    pred_mask, canopy_name, canopy_mae = lodo_best_model(X_canopy, y_mask, days, seed, tune)
    canopy_persist = float(np.nanmean(np.abs(y_mask - usable["pheno_mask_t"].to_numpy(dtype=float))))
    print(f"  canopy: {canopy_name} MAE={canopy_mae:.1f}px  (persist {canopy_persist:.1f})")

    pred_fw_direct, direct_name, mae_d = lodo_best_model(X_direct, y_fw, days, seed, tune)
    ok = np.isfinite(pred_fw_direct)
    mae_d, rmse_d, r2_d = metrics(y_fw[ok], pred_fw_direct[ok])
    print(f"  direct FW: {direct_name} MAE={mae_d:.2f}g")

    # Phenotype bridge
    usable = usable.copy()
    usable["day_future_feat"] = usable["day_future"].astype(float)
    bridge_cols = [c for c in (SENSOR_COLS + DIRECT_EXTRA + ["day_future_feat"]) if c in usable.columns]
    X_bt_df = usable[bridge_cols].copy()
    X_bt_df["pheno_mask_future_in"] = y_mask
    X_bp_df = usable[bridge_cols].copy()
    X_bp_df["pheno_mask_future_in"] = pred_mask
    X_bt = X_bt_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X_bp = X_bp_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Bridge model selection on true phenotype (train), apply on predicted
    zoo = dict(model_zoo(seed))
    pred_fw_pheno = np.full(len(y_fw), np.nan, dtype=float)
    bridge_chosen = []
    for d in np.unique(days):
        te = days == d
        tr = ~te
        if tr.sum() < 5:
            continue
        if tune and tr.sum() >= 8:
            best_name, best_mae = "ElasticNet", float("inf")
            dtr = days[tr]
            for name, model in zoo.items():
                ip = np.full(tr.sum(), np.nan)
                Xtr, ytr = X_bt[tr], y_fw[tr]
                for d2 in np.unique(dtr):
                    te2 = dtr == d2
                    tr2 = ~te2
                    if tr2.sum() < 5:
                        continue
                    ip[te2] = clone_fit_predict(model, Xtr[tr2], ytr[tr2], Xtr[te2])
                mae = float(np.nanmean(np.abs(ytr - ip)))
                if mae < best_mae:
                    best_mae, best_name = mae, name
        else:
            best_name = "ElasticNet"
        bridge_chosen.append(best_name)
        pred_fw_pheno[te] = clone_fit_predict(zoo[best_name], X_bt[tr], y_fw[tr], X_bp[te])

    from collections import Counter

    bridge_name = Counter(bridge_chosen).most_common(1)[0][0] if bridge_chosen else "ElasticNet"
    ok_p = np.isfinite(pred_fw_pheno)
    mae_p, rmse_p, r2_p = metrics(y_fw[ok_p], pred_fw_pheno[ok_p])
    both = ok & ok_p
    cons = float(np.mean(np.abs(pred_fw_direct[both] - pred_fw_pheno[both]))) if both.any() else float("nan")
    print(f"  phenotype FW: {bridge_name}←{canopy_name} MAE={mae_p:.2f}g  |Δ|={cons:.2f}")

    mae_g, rmse_g, r2_g = metrics(y_fw, grow)
    results = [
        RowResult(
            seed,
            "persist",
            "Persist",
            "lodo_target_day",
            len(usable),
            0,
            persist_mae,
            float(math.sqrt(float(np.mean((y_fw - fw_t) ** 2)))),
            _r2(y_fw, fw_t),
            0.0,
            canopy_persist,
            float("nan"),
        ),
        RowResult(
            seed,
            "grow_mean",
            "GrowMean",
            "lodo_target_day",
            len(usable),
            0,
            mae_g,
            rmse_g,
            r2_g,
            persist_mae - mae_g,
            float("nan"),
            float("nan"),
        ),
        RowResult(
            seed,
            "direct_fw",
            direct_name,
            "lodo_target_day",
            int(ok.sum()),
            len(direct_cols),
            mae_d,
            rmse_d,
            r2_d,
            persist_mae - mae_d,
            float("nan"),
            cons,
        ),
        RowResult(
            seed,
            "phenotype_bridge",
            f"{bridge_name}←{canopy_name}",
            "lodo_target_day",
            int(ok_p.sum()),
            len(bridge_cols) + 1,
            mae_p,
            rmse_p,
            r2_p,
            persist_mae - mae_p,
            canopy_mae,
            cons,
        ),
        RowResult(
            seed,
            "canopy_only",
            canopy_name,
            "lodo_target_day",
            len(usable),
            len(canopy_cols),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            canopy_mae,
            float("nan"),
        ),
    ]

    pred_df = usable[
        ["trial", "plant_id", "day_t", "day_future", "fw_t", "fw_future", "pheno_mask_t", "pheno_mask_future"]
    ].copy()
    pred_df["seed"] = seed
    pred_df["pred_mask_future"] = pred_mask
    pred_df["pred_fw_direct"] = pred_fw_direct
    pred_df["pred_fw_phenotype"] = pred_fw_pheno
    pred_df["pred_fw_persist"] = fw_t
    pred_df["pred_fw_growmean"] = grow
    return results, pred_df


def aggregate(results: list[RowResult]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in results])
    rows = []
    for key, g in df.groupby(["route", "model", "split"], sort=True):
        rec = dict(zip(["route", "model", "split"], key))
        rec["n_seeds"] = len(g)
        for col in ("mae_g", "rmse_g", "r2", "mae_vs_persist", "canopy_mae_px", "consistency_mae_g"):
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                rec[f"{col}_mean"] = float("nan")
                rec[f"{col}_sd"] = float("nan")
                rec[f"{col}_mean_pm_sd"] = "—"
            elif len(vals) == 1:
                rec[f"{col}_mean"] = float(vals[0])
                rec[f"{col}_sd"] = 0.0
                rec[f"{col}_mean_pm_sd"] = f"{vals[0]:.2f}"
            else:
                rec[f"{col}_mean"] = float(vals.mean())
                rec[f"{col}_sd"] = float(vals.std(ddof=1))
                rec[f"{col}_mean_pm_sd"] = f"{vals.mean():.2f} ± {vals.std(ddof=1):.2f}"
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--tune", action="store_true", help="Nested LODO model selection per fold")
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output dir under ml/ (default aim2_phenotype_tuned if --tune else aim2_phenotype)",
    )
    args = p.parse_args()

    out_name = args.out or ("aim2_phenotype_tuned" if args.tune else "aim2_phenotype")
    out_dir = ML_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Aim 2 Phase A — phenotype-consistency (tuned run)" if args.tune else "Aim 2 Phase A")
    print(f"Seeds={args.seeds}  tune={args.tune}  out={out_dir}")
    pairs = build_pairs()
    pairs.to_csv(out_dir / "plant_pairs_aim2.csv", index=False)
    print(f"  pairs={len(pairs)}  with future mask={pairs['pheno_mask_future'].notna().sum()}")

    all_results: list[RowResult] = []
    pred_frames = []
    for seed in args.seeds:
        res, pred = run_seed(pairs, seed, tune=args.tune)
        all_results.extend(res)
        pred_frames.append(pred)

    res_df = pd.DataFrame([asdict(r) for r in all_results])
    res_df.to_csv(out_dir / "results_aim2_per_seed.csv", index=False)
    agg = aggregate(all_results)
    agg.to_csv(out_dir / "results_aim2_mean_sd.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(out_dir / "predictions_aim2.csv", index=False)

    summary = {"horizon_days": HORIZON, "seeds": args.seeds, "tune": args.tune, "n_pairs": int(len(pairs)), "headline": {}}
    for _, r in agg.iterrows():
        summary["headline"][r["route"]] = {
            "model": r["model"],
            "mae_g_mean_pm_sd": r.get("mae_g_mean_pm_sd"),
            "canopy_mae_px_mean_pm_sd": r.get("canopy_mae_px_mean_pm_sd"),
            "mae_vs_persist_mean": r.get("mae_vs_persist_mean"),
        }
    (out_dir / "aim2_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== HEADLINE (mean ± SD) ===")
    for route in ("persist", "grow_mean", "direct_fw", "phenotype_bridge", "canopy_only"):
        sub = agg[agg["route"] == route]
        if sub.empty:
            continue
        # if multiple models aggregated separately, show best mae
        sub = sub.sort_values("mae_g_mean", na_position="last")
        r = sub.iloc[0]
        print(
            f"  {route:18s}  {str(r['model']):22s}  "
            f"FW={r['mae_g_mean_pm_sd']:14s}  canopy_px={r['canopy_mae_px_mean_pm_sd']}"
        )
    print(f"\nWrote under {out_dir}/")


if __name__ == "__main__":
    main()

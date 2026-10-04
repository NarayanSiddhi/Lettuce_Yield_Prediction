#!/usr/bin/env python3
"""
Aim 1 FW methodology refine (Lin / Zhang / FGTD inspired).

Why: Pix2Pix needs thousands of pairs; we have 79 plant-days. Literature that
fits this n uses compact canopy + climate features → tree/linear regressors
(Lin multimodal traits; FGTD → XGBoost downstream; Zhang physiology priors).

Refinements vs locked Aim 1 ablation:
  1. Curated low-dim feature packs (lin_phy ≪ 163 compact dims)
  2. Nested LODO model (+ small XGB HP) selection per outer fold
  3. Optional log1p(FW) target with expm1 back-transform
  4. Headline = LODO only (LOO/LOPO still reported as optimistic checks)
  5. Multi-seed mean±SD

Take: Lin-style trait fusion; FGTD sparse-label XGBoost; Aim2 nested LODO.
Skip: full image generation; cloning Zhang physiology ODE.
Improve: fewer features + proper nested selection for n=79.

Usage:
  python3 ml/refine_aim1_fw.py --seeds 42,43,44,45,46 --tune
  python3 ml/refine_aim1_fw.py --seeds 42 --packs lin_phy,compact --no-tune
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
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

PROJECT = Path(__file__).resolve().parents[1]
PLANT = Path(__file__).resolve().parent / "plant"
OUT = PLANT / "fw_refine"
SOURCE = PLANT / "plant_checkpoint_trial1_compact.csv"
TARGET = "target_fw_g"
META_DROP = {
    "sample_id",
    "trial",
    "date",
    "window_start",
    "window_end",
    "label_source",
    "tray_median_fw_g",
    TARGET,
}


@dataclass
class Row:
    seed: int
    pack: str
    split: str
    model: str
    n_features: int
    n_rows: int
    log_target: bool
    tune: bool
    mae_g: float
    rmse_g: float
    r2: float
    day_median_mae_g: float


def metrics(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    mae = float(np.mean(np.abs(y - pred)))
    rmse = float(math.sqrt(float(np.mean((y - pred) ** 2))))
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else float("inf"))
    return mae, rmse, r2


def day_median_mae(days: np.ndarray, y: np.ndarray, pred: np.ndarray) -> float:
    err = []
    for d in np.unique(days):
        m = days == d
        err.append(abs(float(np.median(pred[m])) - float(np.median(y[m]))))
    return float(np.mean(err)) if err else float("nan")


def select_lin_phy(cols: list[str]) -> list[str]:
    """
    Lin-inspired compact canopy + climate/nutrient means (+ DAT, plant_id).
    Prefer *_mean_wmean / plant_img_* / win_* slopes; drop most actuator noise.
    """
    keep: list[str] = []
    must = ["day", "plant_id", "window_days_count"]
    for c in must:
        if c in cols:
            keep.append(c)

    plant_img = [c for c in cols if c.startswith("plant_img_")]
    keep.extend(sorted(plant_img))

    canopy_exact = [
        "img_mask_area_mean_wmean",
        "img_mask_area_sum_wmean",
        "img_mask_area_max_wmean",
        "img_coverage_frac_wmean",
        "img_plant_color_mean_wmean",
        "img_blur_mean_wmean",
        "win_img_mask_area_sum_last",
        "win_img_mask_area_sum_delta",
        "win_img_mask_area_sum_slope",
        "win_img_mask_area_sum_pct_delta",
        "win_img_coverage_frac_last",
        "win_img_coverage_frac_delta",
        "win_img_coverage_frac_slope",
        "win_img_plant_color_mean_last",
        "win_img_plant_color_mean_delta",
        "win_img_plant_color_mean_slope",
    ]
    for c in canopy_exact:
        if c in cols and c not in keep:
            keep.append(c)

    climate_patterns = [
        r"^env_day_air_temp_c_mean_wmean$",
        r"^env_night_air_temp_c_mean_wmean$",
        r"^env_day_humidity_pct_mean_wmean$",
        r"^env_night_humidity_pct_mean_wmean$",
        r"^env_day_co2_ppm_mean_wmean$",
        r"^env_night_co2_ppm_mean_wmean$",
        r"^env_day_vpd_pa_mean_wmean$",
        r"^env_night_vpd_pa_mean_wmean$",
        r"^nutrient_ec_us_cm_mean_wmean$",
        r"^nutrient_ph_mean_wmean$",
        r"^nutrient_tds_ppm_mean_wmean$",
        r"^env_act_day_light_duration_s_sum_wmean$",
    ]
    for pat in climate_patterns:
        for c in cols:
            if re.match(pat, c) and c not in keep:
                keep.append(c)

    # physiology-lite engineered (added later as columns)
    for c in ("day_sq", "log_mask_area", "area_x_day"):
        if c in cols and c not in keep:
            keep.append(c)
    return keep


def select_canopy(cols: list[str]) -> list[str]:
    base = ["day", "plant_id"]
    hits = [
        c
        for c in cols
        if c.startswith("plant_img_")
        or c.startswith("img_mask_")
        or c.startswith("img_coverage_")
        or c.startswith("img_plant_color_")
        or c.startswith("win_img_")
        or c in ("day_sq", "log_mask_area", "area_x_day")
    ]
    return [c for c in base + sorted(hits) if c in cols]


def select_climate(cols: list[str]) -> list[str]:
    base = ["day", "plant_id", "day_sq"]
    hits = []
    for c in cols:
        if c in ("day_sq",):
            hits.append(c)
            continue
        if c.startswith(("env_day_", "env_night_")) and c.endswith("_mean_wmean"):
            hits.append(c)
        elif c.startswith("nutrient_") and "act_" not in c and c.endswith("_mean_wmean"):
            hits.append(c)
        elif c == "env_act_day_light_duration_s_sum_wmean":
            hits.append(c)
    return [c for c in base + sorted(set(hits)) if c in cols]


def pack_columns(pack: str, all_feat_cols: list[str]) -> list[str]:
    if pack == "compact":
        return list(all_feat_cols)
    if pack == "lin_phy":
        return select_lin_phy(all_feat_cols)
    if pack == "canopy":
        return select_canopy(all_feat_cols)
    if pack == "climate":
        return select_climate(all_feat_cols)
    raise ValueError(pack)


def load_table() -> tuple[pd.DataFrame, list[str]]:
    if not SOURCE.exists():
        raise SystemExit(f"Missing {SOURCE}. Run: python3 export_plant_ml_datasets.py")
    df = pd.read_csv(SOURCE).copy()
    # physiology-lite extras (Zhang-inspired growth-stage / allometry proxies)
    area = pd.to_numeric(df.get("plant_img_mask_area_px_last", np.nan), errors="coerce")
    if area.isna().all():
        area = pd.to_numeric(df.get("img_mask_area_mean_wmean", np.nan), errors="coerce")
    day = pd.to_numeric(df["day"], errors="coerce")
    df["day_sq"] = day ** 2
    df["log_mask_area"] = np.log1p(area.clip(lower=0))
    df["area_x_day"] = area * day

    feat_cols = [c for c in df.columns if c not in META_DROP]
    # numeric only
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, feat_cols


def zoo(seed: int) -> list[tuple[str, object]]:
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
                    ("model", ElasticNet(alpha=0.05, l1_ratio=0.25, random_state=seed, max_iter=50_000)),
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
                            n_estimators=400, max_depth=4, random_state=seed, n_jobs=1
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
                            n_estimators=300, max_depth=2, random_state=seed
                        ),
                    ),
                ]
            ),
        ),
    ]
    try:
        from xgboost import XGBRegressor

        for tag, kwargs in (
            ("XGBoost", dict(n_estimators=120, max_depth=3, learning_rate=0.08, reg_lambda=2.0)),
            ("XGBoost_d2", dict(n_estimators=200, max_depth=2, learning_rate=0.05, reg_lambda=5.0)),
            ("XGBoost_d4", dict(n_estimators=80, max_depth=4, learning_rate=0.1, reg_lambda=1.0)),
        ):
            models.append(
                (
                    tag,
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            (
                                "model",
                                XGBRegressor(
                                    subsample=0.9,
                                    colsample_bytree=0.8,
                                    random_state=seed,
                                    n_jobs=1,
                                    **kwargs,
                                ),
                            ),
                        ]
                    ),
                )
            )
    except Exception:
        pass
    return models


def fit_predict(model, Xtr, ytr, Xte, log_target: bool) -> np.ndarray:
    m = clone(model)
    if log_target:
        m.fit(Xtr, np.log1p(np.clip(ytr, 0, None)))
        return np.expm1(m.predict(Xte)).astype(float)
    m.fit(Xtr, ytr)
    return m.predict(Xte).astype(float)


def lodo_fixed(model, X, y, days, log_target: bool) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    for d in np.unique(days):
        te = days == d
        pred[te] = fit_predict(model, X[~te], y[~te], X[te], log_target)
    return pred


def lodo_nested(zoo_models, X, y, days, log_target: bool) -> tuple[np.ndarray, list[str]]:
    """Per outer day: pick model by inner LODO MAE on remaining days."""
    pred = np.empty(len(y), dtype=float)
    chosen: list[str] = []
    for d in np.unique(days):
        te = days == d
        tr = ~te
        Xtr, ytr, dtr = X[tr], y[tr], days[tr]
        best_name, best_mae = zoo_models[0][0], float("inf")
        uniq_tr = np.unique(dtr)
        if len(uniq_tr) >= 3:
            for name, model in zoo_models:
                inner = np.full(len(ytr), np.nan)
                for d2 in uniq_tr:
                    te2 = dtr == d2
                    tr2 = ~te2
                    if tr2.sum() < 5:
                        continue
                    inner[te2] = fit_predict(model, Xtr[tr2], ytr[tr2], Xtr[te2], log_target)
                mae = float(np.nanmean(np.abs(ytr - inner)))
                if mae < best_mae:
                    best_mae, best_name = mae, name
        else:
            # too few days: fall back to ElasticNet / first
            best_name = "ElasticNet" if any(n == "ElasticNet" for n, _ in zoo_models) else zoo_models[0][0]
        model = dict(zoo_models)[best_name]
        pred[te] = fit_predict(model, Xtr, ytr, X[te], log_target)
        chosen.append(f"day{int(d)}:{best_name}")
    return pred, chosen


def loo_predict(model, X, y, log_target: bool) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        pred[i] = fit_predict(model, X[mask], y[mask], X[i : i + 1], log_target)[0]
    return pred


def lopo_predict(model, X, y, plants, log_target: bool) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    for p in np.unique(plants):
        te = plants == p
        pred[te] = fit_predict(model, X[~te], y[~te], X[te], log_target)
    return pred


def run_pack(
    df: pd.DataFrame,
    feat_cols: list[str],
    pack: str,
    seed: int,
    tune: bool,
    log_target: bool,
    splits: list[str],
) -> tuple[list[Row], pd.DataFrame]:
    cols = pack_columns(pack, feat_cols)
    if not cols:
        raise SystemExit(f"Pack {pack} selected 0 features")
    X = df[cols].to_numpy(dtype=float)
    y = pd.to_numeric(df[TARGET], errors="coerce").to_numpy(dtype=float)
    days = pd.to_numeric(df["day"], errors="coerce").to_numpy(dtype=int)
    plants = pd.to_numeric(df["plant_id"], errors="coerce").to_numpy(dtype=int)
    models = zoo(seed)

    rows: list[Row] = []
    pred_frames = []

    print(
        f"\n=== seed={seed} pack={pack} n_feat={len(cols)} tune={tune} log_target={log_target} ===",
        flush=True,
    )

    # LODO (primary)
    if "lodo" in splits:
        if tune:
            pred, chosen = lodo_nested(models, X, y, days, log_target)
            model_label = "NestedSelect"
            print(f"  LODO nested choices: {chosen}", flush=True)
        else:
            # evaluate all; keep best for headline row + save all
            best_pred, best_name, best_mae = None, "", float("inf")
            for name, model in models:
                pred = lodo_fixed(model, X, y, days, log_target)
                mae, rmse, r2 = metrics(y, pred)
                dmed = day_median_mae(days, y, pred)
                rows.append(
                    Row(seed, pack, "lodo", name, len(cols), len(y), log_target, tune, mae, rmse, r2, dmed)
                )
                print(f"  LODO {name:12s} MAE={mae:6.2f} R2={r2:5.3f}", flush=True)
                if mae < best_mae:
                    best_mae, best_name, best_pred = mae, name, pred
            pred, model_label = best_pred, f"Best:{best_name}"
        mae, rmse, r2 = metrics(y, pred)
        dmed = day_median_mae(days, y, pred)
        rows.append(
            Row(seed, pack, "lodo", model_label, len(cols), len(y), log_target, tune, mae, rmse, r2, dmed)
        )
        print(f"  LODO headline {model_label} MAE={mae:6.2f} R2={r2:5.3f} dayMed={dmed:6.2f}", flush=True)
        pf = df[["sample_id", "trial", "day", "plant_id", "date"]].copy()
        pf["split"] = "lodo"
        pf["pack"] = pack
        pf["seed"] = seed
        pf["model"] = model_label
        pf[TARGET] = y
        pf["pred_fw_g"] = pred
        pf["abs_error_g"] = np.abs(pred - y)
        pred_frames.append(pf)

    # Optimistic checks: fixed best XGBoost or ElasticNet
    default_model = dict(models).get("XGBoost") or dict(models)["ElasticNet"]
    default_name = "XGBoost" if "XGBoost" in dict(models) else "ElasticNet"
    if "loo" in splits:
        pred = loo_predict(default_model, X, y, log_target)
        mae, rmse, r2 = metrics(y, pred)
        dmed = day_median_mae(days, y, pred)
        rows.append(Row(seed, pack, "loo", default_name, len(cols), len(y), log_target, False, mae, rmse, r2, dmed))
        print(f"  LOO  {default_name:12s} MAE={mae:6.2f} (optimistic)", flush=True)
    if "lopo" in splits:
        pred = lopo_predict(default_model, X, y, plants, log_target)
        mae, rmse, r2 = metrics(y, pred)
        dmed = day_median_mae(days, y, pred)
        rows.append(Row(seed, pack, "lopo", default_name, len(cols), len(y), log_target, False, mae, rmse, r2, dmed))
        print(f"  LOPO {default_name:12s} MAE={mae:6.2f} (optimistic)", flush=True)

    # Mean baseline LODO
    if "lodo" in splits:
        pred_b = np.empty_like(y)
        for d in np.unique(days):
            te = days == d
            pred_b[te] = float(np.mean(y[~te]))
        mae, rmse, r2 = metrics(y, pred_b)
        dmed = day_median_mae(days, y, pred_b)
        rows.append(Row(seed, pack, "lodo", "MeanBaseline", 0, len(y), False, False, mae, rmse, r2, dmed))

    preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    return rows, preds


def aggregate(rows: list[Row]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in rows])
    # headline: LODO NestedSelect or Best:* only
    h = df[df["split"] == "lodo"].copy()
    h = h[h["model"].str.startswith("Nested") | h["model"].str.startswith("Best") | (h["model"] == "MeanBaseline")]
    gcols = ["pack", "split", "model", "log_target", "tune", "n_features"]
    agg = (
        h.groupby(gcols, as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mae_mean=("mae_g", "mean"),
            mae_sd=("mae_g", "std"),
            rmse_mean=("rmse_g", "mean"),
            rmse_sd=("rmse_g", "std"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            day_med_mae_mean=("day_median_mae_g", "mean"),
            day_med_mae_sd=("day_median_mae_g", "std"),
        )
    )
    agg["mae_pm"] = agg.apply(
        lambda r: f"{r.mae_mean:.2f} ± {(0.0 if pd.isna(r.mae_sd) else r.mae_sd):.2f}", axis=1
    )
    return agg.sort_values(["split", "mae_mean"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--packs", default="lin_phy,canopy,climate,compact")
    p.add_argument("--tune", action="store_true", help="Nested LODO model selection")
    p.add_argument("--no-tune", action="store_true")
    p.add_argument("--log-target", action="store_true", help="Also run log1p(FW) variant")
    p.add_argument("--splits", default="lodo,loo,lopo")
    args = p.parse_args()

    tune = True if args.tune else False
    if args.no_tune:
        tune = False
    if not args.tune and not args.no_tune:
        tune = True  # default: nested on

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    packs = [s.strip() for s in args.packs.split(",") if s.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    log_flags = [False, True] if args.log_target else [False]

    OUT.mkdir(parents=True, exist_ok=True)
    df, feat_cols = load_table()

    # document lin_phy columns
    lin_cols = pack_columns("lin_phy", feat_cols)
    (OUT / "lin_phy_features.json").write_text(
        json.dumps({"n": len(lin_cols), "columns": lin_cols}, indent=2)
    )

    all_rows: list[Row] = []
    all_preds = []
    for seed in seeds:
        for pack in packs:
            for log_t in log_flags:
                rows, preds = run_pack(df, feat_cols, pack, seed, tune, log_t, splits)
                all_rows.extend(rows)
                if len(preds):
                    all_preds.append(preds)

    raw = pd.DataFrame([asdict(r) for r in all_rows])
    raw.to_csv(OUT / "fw_refine_all_rows.csv", index=False)
    if all_preds:
        pd.concat(all_preds, ignore_index=True).to_csv(OUT / "fw_refine_predictions_lodo.csv", index=False)

    summary = aggregate(all_rows)
    summary.to_csv(OUT / "fw_refine_summary_lodo.csv", index=False)

    # best pack headline
    cand = summary[(summary["model"] != "MeanBaseline") & (summary["split"] == "lodo")]
    print("\n=== LODO headline (mean±SD over seeds) ===", flush=True)
    if len(cand):
        print(cand.to_string(index=False), flush=True)
        best = cand.iloc[0]
        meta = {
            "primary_split": "lodo",
            "best_pack": best["pack"],
            "best_model": best["model"],
            "mae_pm": best["mae_pm"],
            "n_features": int(best["n_features"]),
            "tune": bool(best["tune"]),
            "log_target": bool(best["log_target"]),
            "baseline_note": "LOO/LOPO are optimistic; LODO is the Aim 1 headline",
            "literature": {
                "take": "Lin canopy+climate traits; FGTD→XGBoost; Aim2 nested LODO",
                "skip": "Chen/Drees-scale image GANs for FW",
                "improve": "curated lin_phy + nested select for n=79",
            },
        }
        (OUT / "fw_refine_headline.json").write_text(json.dumps(meta, indent=2))
        print(f"\nHeadline → {meta}", flush=True)
    print(f"Wrote {OUT}/", flush=True)


if __name__ == "__main__":
    main()

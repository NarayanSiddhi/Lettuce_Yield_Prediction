"""
Canopy forecast (Step 2): features at day t → canopy size at day t+4.

Primary target: FastSAM mask_area_px (cup-level) / img_mask_area_sum (tray-daily).
Uses only information available at time t (no future climate leakage).

Baselines:
  Persist  — same canopy size in 4 days
  GrowMean — current + mean 4-day gain from other folds

Usage:
  python3 ml/train_canopy_forecast.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

ML_DIR = Path(__file__).resolve().parent
PROJECT = ML_DIR.parent
OUT_DIR = ML_DIR / "forecast_canopy"
HORIZON = 4

CLIMATE_COLS = [
    "T_air_C",
    "RH_pct",
    "CO2_ppm",
    "EC_mS_cm",
    "PPFD_umol_m2_s",
    "DLI_mol_m2_d",
    "photoperiod_h",
]


@dataclass
class Result:
    level: str
    model: str
    split: str
    n_rows: int
    n_features: int
    mae: float
    rmse: float
    r2: float
    mae_vs_persist: float
    target: str


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


def load_drivers(trial: str) -> pd.DataFrame:
    path = PROJECT / "Lettuce_model" / "data" / f"{trial}_daily_drivers.csv"
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    keep = ["DAT", "date"] + [c for c in CLIMATE_COLS if c in d.columns]
    return d[keep].sort_values("DAT")


def climate_history_features(drivers: pd.DataFrame, dat: int, window: int = 4) -> dict:
    """Means over DAT in [dat-window+1, dat], using only past/present."""
    sub = drivers[(drivers["DAT"] >= dat - window + 1) & (drivers["DAT"] <= dat)]
    out = {}
    for c in CLIMATE_COLS:
        if c not in sub.columns:
            continue
        out[f"clim_{c}_now"] = float(drivers.loc[drivers["DAT"] == dat, c].iloc[0]) if (drivers["DAT"] == dat).any() else np.nan
        out[f"clim_{c}_mean{window}d"] = float(sub[c].mean()) if len(sub) else np.nan
    return out


def build_cup_pairs() -> tuple[pd.DataFrame, list[str]]:
    cup = pd.read_csv(PROJECT / "segmentation" / "daily_cup_features.csv")
    cup["date"] = pd.to_datetime(cup["date"]).dt.normalize()
    # one row per trial/cup/date (already mostly unique)
    cup = (
        cup.sort_values(["trial", "cup_id", "date", "selection_score"], ascending=[True, True, True, False])
        .groupby(["trial", "cup_id", "date"], as_index=False)
        .first()
    )

    rows = []
    for trial, g_trial in cup.groupby("trial"):
        drivers = load_drivers(trial)
        g_trial = g_trial.merge(drivers[["date", "DAT"]], on="date", how="inner")
        for cup_id, g in g_trial.groupby("cup_id"):
            by_dat = {int(r["DAT"]): r for _, r in g.iterrows()}
            for dat_t, row_t in by_dat.items():
                dat_f = dat_t + HORIZON
                if dat_f not in by_dat:
                    continue
                row_f = by_dat[dat_f]
                feat = {
                    "trial": trial,
                    "cup_id": int(cup_id),
                    "day_t": dat_t,
                    "day_future": dat_f,
                    "mask_area_t": float(row_t["mask_area_px"]),
                    "mask_area_future": float(row_f["mask_area_px"]),
                    "color_t": float(row_t["plant_color_frac"]),
                    "color_future": float(row_f["plant_color_frac"]),
                    "blur_t": float(row_t["blur_score"]),
                }
                prev = dat_t - HORIZON
                if prev in by_dat:
                    feat["mask_area_prev"] = float(by_dat[prev]["mask_area_px"])
                    feat["gain_prev_4d"] = feat["mask_area_t"] - feat["mask_area_prev"]
                else:
                    feat["mask_area_prev"] = np.nan
                    feat["gain_prev_4d"] = np.nan
                feat.update(climate_history_features(drivers, dat_t))
                rows.append(feat)

    out = pd.DataFrame(rows)
    meta = {
        "trial",
        "cup_id",
        "day_t",
        "day_future",
        "mask_area_future",
        "color_future",
    }
    feature_cols = [c for c in out.columns if c not in meta]
    # drop constant / all-nan
    keep = []
    for c in feature_cols:
        s = pd.to_numeric(out[c], errors="coerce")
        if s.notna().sum() == 0 or s.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return out, keep


def build_tray_daily_pairs() -> tuple[pd.DataFrame, list[str]]:
    rows = []
    for trial in ("trial1", "trial2"):
        img_path = PROJECT / f"image_features_daily_{trial}.csv"
        if not img_path.exists():
            continue
        img = pd.read_csv(img_path)
        img["date"] = pd.to_datetime(img["date"]).dt.normalize()
        drivers = load_drivers(trial)
        img = img.merge(drivers[["date", "DAT"]], on="date", how="inner")
        by_dat = {int(r["DAT"]): r for _, r in img.iterrows()}
        for dat_t, row_t in by_dat.items():
            dat_f = dat_t + HORIZON
            if dat_f not in by_dat:
                continue
            row_f = by_dat[dat_f]
            feat = {
                "trial": trial,
                "cup_id": -1,
                "day_t": dat_t,
                "day_future": dat_f,
                "mask_area_t": float(row_t["img_mask_area_sum"]),
                "mask_area_future": float(row_f["img_mask_area_sum"]),
                "coverage_t": float(row_t["img_coverage_frac"]),
                "coverage_future": float(row_f["img_coverage_frac"]),
                "color_t": float(row_t["img_plant_color_mean"]),
            }
            prev = dat_t - HORIZON
            if prev in by_dat:
                feat["mask_area_prev"] = float(by_dat[prev]["img_mask_area_sum"])
                feat["gain_prev_4d"] = feat["mask_area_t"] - feat["mask_area_prev"]
            else:
                feat["mask_area_prev"] = np.nan
                feat["gain_prev_4d"] = np.nan
            feat.update(climate_history_features(drivers, dat_t))
            rows.append(feat)

    out = pd.DataFrame(rows)
    meta = {"trial", "cup_id", "day_t", "day_future", "mask_area_future", "coverage_future"}
    feature_cols = []
    for c in out.columns:
        if c in meta:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        if s.notna().sum() == 0 or s.nunique(dropna=True) <= 1:
            continue
        feature_cols.append(c)
    return out, feature_cols


def build_regressors(random_state: int = 42):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

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
                    ("model", ElasticNet(alpha=0.05, l1_ratio=0.25, max_iter=50_000, random_state=random_state)),
                ]
            ),
        ),
        (
            "SVR-RBF",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", SVR(C=10.0, gamma="scale")),
                ]
            ),
        ),
        (
            "RandomForest",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("model", RandomForestRegressor(n_estimators=200, max_depth=4, random_state=random_state)),
                ]
            ),
        ),
        (
            "GradBoost",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("model", GradientBoostingRegressor(n_estimators=120, max_depth=2, random_state=random_state)),
                ]
            ),
        ),
    ]
    try:
        from xgboost import XGBRegressor  # type: ignore

        models.append(
            (
                "XGBoost",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        (
                            "model",
                            XGBRegressor(
                                n_estimators=100,
                                max_depth=3,
                                learning_rate=0.08,
                                subsample=0.9,
                                colsample_bytree=0.8,
                                reg_lambda=2.0,
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
            )
        )
    except Exception:
        pass
    return models


def loo_groups(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if tr.sum() == 0:
            pred[te] = float(np.mean(y))
            continue
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te]).astype(float)
    return pred


def baseline_persist(x_t: np.ndarray) -> np.ndarray:
    return x_t.copy()


def baseline_grow_mean(x_t: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    gain = y - x_t
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        mean_gain = float(np.mean(gain[tr])) if tr.any() else float(np.mean(gain))
        pred[te] = x_t[te] + mean_gain
    return pred


def run_one(
    level: str,
    table: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
    x_t_col: str,
    split_specs: list[tuple[str, np.ndarray]],
) -> tuple[list[Result], list[dict]]:
    results: list[Result] = []
    pred_rows: list[dict] = []

    X = table[feat_cols].to_numpy(dtype=float)
    y = table[target_col].to_numpy(dtype=float)
    x_t = table[x_t_col].to_numpy(dtype=float)

    for split_name, groups in split_specs:
        persist = baseline_persist(x_t)
        persist_mae, _, _ = metrics(y, persist)

        for bname, bpred in (
            ("Persist", persist),
            ("GrowMean", baseline_grow_mean(x_t, y, groups)),
        ):
            mae, rmse, r2 = metrics(y, bpred)
            results.append(
                Result(level, bname, split_name, len(y), 0, mae, rmse, r2, persist_mae - mae, target_col)
            )
            print(
                f"  [{level}/{split_name}] {bname:12s} MAE={mae:.1f}  R²={r2:.3f}",
                flush=True,
            )
            for i in range(len(y)):
                pred_rows.append(
                    {
                        "level": level,
                        "model": bname,
                        "split": split_name,
                        "target": target_col,
                        "trial": table.iloc[i]["trial"],
                        "cup_id": int(table.iloc[i]["cup_id"]),
                        "day_t": int(table.iloc[i]["day_t"]),
                        "day_future": int(table.iloc[i]["day_future"]),
                        "y_t": float(x_t[i]),
                        "y_future": float(y[i]),
                        "pred": float(bpred[i]),
                        "abs_err": float(abs(bpred[i] - y[i])),
                    }
                )

        for name, model in build_regressors():
            pred = loo_groups(model, X, y, groups)
            mae, rmse, r2 = metrics(y, pred)
            results.append(
                Result(level, name, split_name, len(y), len(feat_cols), mae, rmse, r2, persist_mae - mae, target_col)
            )
            print(
                f"  [{level}/{split_name}] {name:12s} MAE={mae:.1f}  R²={r2:.3f}  "
                f"(vs persist {persist_mae - mae:+.1f})",
                flush=True,
            )
            for i in range(len(y)):
                pred_rows.append(
                    {
                        "level": level,
                        "model": name,
                        "split": split_name,
                        "target": target_col,
                        "trial": table.iloc[i]["trial"],
                        "cup_id": int(table.iloc[i]["cup_id"]),
                        "day_t": int(table.iloc[i]["day_t"]),
                        "day_future": int(table.iloc[i]["day_future"]),
                        "y_t": float(x_t[i]),
                        "y_future": float(y[i]),
                        "pred": float(pred[i]),
                        "abs_err": float(abs(pred[i] - y[i])),
                    }
                )

    return results, pred_rows


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="all", choices=["all", "cup", "tray"])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[Result] = []
    all_preds: list[dict] = []

    levels = ["cup", "tray"] if args.level == "all" else [args.level]

    if "cup" in levels:
        table, feat_cols = build_cup_pairs()
        table.to_csv(OUT_DIR / "cup_pairs_mask_area.csv", index=False)
        print(f"Cup pairs: {len(table)} rows, {len(feat_cols)} features", flush=True)
        # group ids: leave-one-cup (within both trials: trial|cup), leave-one-trial, lodo target day
        cup_group = (table["trial"].astype(str) + "_" + table["cup_id"].astype(str)).to_numpy()
        splits = [
            ("loo_cup", cup_group),
            ("loo_trial", table["trial"].to_numpy()),
            ("lodo_target_day", table["day_future"].to_numpy()),
        ]
        res, preds = run_one("cup", table, feat_cols, "mask_area_future", "mask_area_t", splits)
        all_results.extend(res)
        all_preds.extend(preds)

    if "tray" in levels:
        table, feat_cols = build_tray_daily_pairs()
        table.to_csv(OUT_DIR / "tray_pairs_mask_area.csv", index=False)
        print(f"Tray daily pairs: {len(table)} rows, {len(feat_cols)} features", flush=True)
        splits = [
            ("loo_trial", table["trial"].to_numpy()),
            ("lodo_target_day", table["day_future"].to_numpy()),
        ]
        res, preds = run_one("tray", table, feat_cols, "mask_area_future", "mask_area_t", splits)
        all_results.extend(res)
        all_preds.extend(preds)

    results_path = OUT_DIR / "results_canopy_forecast.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "level",
                "model",
                "split",
                "target",
                "n_rows",
                "n_features",
                "mae",
                "rmse",
                "r2",
                "mae_vs_persist",
            ],
        )
        w.writeheader()
        for r in all_results:
            w.writerow(
                {
                    "level": r.level,
                    "model": r.model,
                    "split": r.split,
                    "target": r.target,
                    "n_rows": r.n_rows,
                    "n_features": r.n_features,
                    "mae": f"{r.mae:.4f}",
                    "rmse": f"{r.rmse:.4f}",
                    "r2": f"{r.r2:.6f}",
                    "mae_vs_persist": f"{r.mae_vs_persist:.4f}",
                }
            )
    print(f"Wrote {results_path}", flush=True)

    pred_path = OUT_DIR / "predictions_canopy_forecast.csv"
    if all_preds:
        with pred_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_preds[0].keys()))
            w.writeheader()
            w.writerows(all_preds)
        print(f"Wrote {pred_path}", flush=True)

    df = pd.DataFrame(
        [
            {
                "level": r.level,
                "model": r.model,
                "split": r.split,
                "mae": r.mae,
                "r2": r.r2,
                "mae_vs_persist": r.mae_vs_persist,
                "n_rows": r.n_rows,
            }
            for r in all_results
        ]
    )
    summary = {"horizon_days": HORIZON, "best": {}}
    for level in df["level"].unique():
        for split in df.loc[df.level == level, "split"].unique():
            sub = df[(df.level == level) & (df.split == split) & (df.model != "Persist")]
            if sub.empty:
                continue
            best = sub.sort_values("mae").iloc[0]
            persist = df[(df.level == level) & (df.split == split) & (df.model == "Persist")]
            summary["best"][f"{level}/{split}"] = {
                "model": best.model,
                "mae": float(best.mae),
                "r2": float(best.r2),
                "mae_vs_persist": float(best.mae_vs_persist),
                "persist_mae": float(persist.iloc[0].mae) if len(persist) else None,
                "n_rows": int(best.n_rows),
            }
    (OUT_DIR / "canopy_forecast_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()

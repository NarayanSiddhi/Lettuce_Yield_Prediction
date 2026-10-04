"""
Next-checkpoint forecast: features at DAT t → fresh weight at DAT t+4.

This is a true forecast (not same-day nowcast). Baselines:
  - Persist: predict current weight
  - GrowMean: current + mean 4-day gain seen in training fold

Usage:
  python3 ml/train_next_checkpoint_forecast.py
  python3 ml/train_next_checkpoint_forecast.py --level plant
  python3 ml/train_next_checkpoint_forecast.py --level tray
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
PLANT_DIR = ML_DIR / "plant"
OUT_DIR = ML_DIR / "forecast_next_checkpoint"
HORIZON = 4

IMAGE_KEYS = [
    "img_mask_area_sum_wmean",
    "img_mask_area_mean_wmean",
    "img_coverage_frac_wmean",
    "img_plant_color_mean_wmean",
    "win_img_mask_area_sum_slope",
    "win_img_mask_area_sum_delta",
    "win_img_coverage_frac_slope",
    "plant_img_mask_area_px_wmean",
    "plant_img_mask_area_px_last",
    "has_plant_cup_image",
]
SENSOR_KEYS = [
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
TRAY_IMAGE_KEYS = [
    "img_mask_area_sum_wmean",
    "img_mask_area_mean_wmean",
    "img_coverage_frac_wmean",
    "img_plant_color_mean_wmean",
    "win_img_mask_area_sum_slope",
    "win_img_mask_area_sum_delta",
    "win_img_coverage_frac_slope",
]
TRAY_SENSOR_KEYS = [
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


@dataclass
class ForecastResult:
    level: str
    variant: str
    model: str
    split: str
    n_rows: int
    n_features: int
    mae_g: float
    rmse_g: float
    r2: float
    mae_vs_persist: float  # positive => beats persist baseline


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


def drop_bad_columns(X: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for c in X.columns:
        s = pd.to_numeric(X[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return X[keep]


def variant_keys(variant: str, level: str) -> list[str]:
    if level == "plant":
        img, sens = IMAGE_KEYS, SENSOR_KEYS
    else:
        img, sens = TRAY_IMAGE_KEYS, TRAY_SENSOR_KEYS
    if variant == "image":
        return list(img)
    if variant == "sensors":
        return list(sens)
    return list(img) + list(sens)


def build_plant_pairs(variant: str) -> tuple[pd.DataFrame, list[str]]:
    path = {
        "compact": PLANT_DIR / "plant_checkpoint_trial1_compact.csv",
        "image": PLANT_DIR / "plant_checkpoint_trial1_image.csv",
        "sensors": PLANT_DIR / "plant_checkpoint_trial1_sensors.csv",
    }[variant]
    df = pd.read_csv(path)
    residual = pd.read_csv(PLANT_DIR / "plant_residual_trial1.csv")[
        ["plant_id", "day", "modelled_g", "target_residual_g"]
    ]
    df = df.merge(residual, on=["plant_id", "day"], how="left")
    keys = [c for c in variant_keys(variant, "plant") if c in df.columns]

    rows = []
    for plant_id, g in df.groupby("plant_id"):
        g = g.sort_values("day")
        by_day = {int(r["day"]): r for _, r in g.iterrows()}
        for day_t, row_t in by_day.items():
            day_f = day_t + HORIZON
            if day_f not in by_day:
                continue
            row_f = by_day[day_f]
            feat: dict = {
                "trial": "trial1",
                "plant_id": int(plant_id),
                "day_t": day_t,
                "day_future": day_f,
                "fw_t": float(row_t["target_fw_g"]),
                "fw_future": float(row_f["target_fw_g"]),
                "modelled_g_t": float(row_t["modelled_g"]) if pd.notna(row_t["modelled_g"]) else np.nan,
                "residual_g_t": float(row_t["target_residual_g"])
                if pd.notna(row_t["target_residual_g"])
                else np.nan,
                "delta_fw": float(row_f["target_fw_g"]) - float(row_t["target_fw_g"]),
            }
            # previous interval gain if available (t-4 → t)
            prev = day_t - HORIZON
            if prev in by_day:
                feat["fw_prev"] = float(by_day[prev]["target_fw_g"])
                feat["gain_prev_4d"] = feat["fw_t"] - feat["fw_prev"]
            else:
                feat["fw_prev"] = np.nan
                feat["gain_prev_4d"] = np.nan
            for c in keys:
                val = row_t[c] if c in row_t.index else np.nan
                feat[c] = float(val) if pd.notna(val) else np.nan
            rows.append(feat)

    out = pd.DataFrame(rows)
    meta = {
        "trial",
        "plant_id",
        "day_t",
        "day_future",
        "fw_future",
        "delta_fw",
    }
    feat_df = drop_bad_columns(out.drop(columns=[c for c in meta if c in out.columns]))
    # always keep fw_t even if somehow dropped
    keep_cols = list(feat_df.columns)
    if "fw_t" not in keep_cols and "fw_t" in out.columns:
        keep_cols = ["fw_t"] + keep_cols
    feature_cols = [c for c in keep_cols if c in out.columns and c not in meta]
    return out, feature_cols


def build_tray_pairs(variant: str) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    keys_wanted = variant_keys(variant, "tray")
    for trial in ("trial1", "trial2"):
        path = ML_DIR / f"checkpoint_residual_{trial}.csv"
        df = pd.read_csv(path)
        keys = [c for c in keys_wanted if c in df.columns]
        by_day = {int(r["day"]): r for _, r in df.iterrows()}
        for day_t, row_t in by_day.items():
            day_f = day_t + HORIZON
            if day_f not in by_day:
                continue
            row_f = by_day[day_f]
            feat: dict = {
                "trial": trial,
                "plant_id": -1,
                "day_t": day_t,
                "day_future": day_f,
                "fw_t": float(row_t["observed_g"]),
                "fw_future": float(row_f["observed_g"]),
                "modelled_g_t": float(row_t["modelled_g"]),
                "residual_g_t": float(row_t["residual_g"]),
                "delta_fw": float(row_f["observed_g"]) - float(row_t["observed_g"]),
            }
            prev = day_t - HORIZON
            if prev in by_day:
                feat["fw_prev"] = float(by_day[prev]["observed_g"])
                feat["gain_prev_4d"] = feat["fw_t"] - feat["fw_prev"]
            else:
                feat["fw_prev"] = np.nan
                feat["gain_prev_4d"] = np.nan
            for c in keys:
                val = row_t[c] if c in row_t.index else np.nan
                feat[c] = float(val) if pd.notna(val) else np.nan
            rows.append(feat)

    out = pd.DataFrame(rows)
    meta = {"trial", "plant_id", "day_t", "day_future", "fw_future", "delta_fw"}
    feat_df = drop_bad_columns(out.drop(columns=[c for c in meta if c in out.columns]))
    feature_cols = [c for c in feat_df.columns if c not in meta]
    if "fw_t" not in feature_cols:
        feature_cols = ["fw_t"] + feature_cols
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
                    ("model", SVR(C=10.0, gamma="scale", epsilon=5.0)),
                ]
            ),
        ),
        (
            "RandomForest",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("model", RandomForestRegressor(n_estimators=200, max_depth=3, random_state=random_state)),
                ]
            ),
        ),
        (
            "GradBoost",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("model", GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=random_state)),
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
                                n_estimators=80,
                                max_depth=2,
                                learning_rate=0.1,
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
    """Leave-one-group-out predictions."""
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


def baseline_persist(fw_t: np.ndarray) -> np.ndarray:
    return fw_t.copy()


def baseline_grow_mean(fw_t: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """current + mean(y - fw_t) from other groups."""
    pred = np.empty(len(y), dtype=float)
    gain = y - fw_t
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        mean_gain = float(np.mean(gain[tr])) if tr.any() else float(np.mean(gain))
        pred[te] = fw_t[te] + mean_gain
    return pred


def run_level(level: str, variants: list[str]) -> tuple[list[ForecastResult], list[dict]]:
    results: list[ForecastResult] = []
    pred_rows: list[dict] = []

    for variant in variants:
        if level == "plant":
            table, feat_cols = build_plant_pairs(variant)
            split_specs = [
                ("loo_plant", table["plant_id"].to_numpy()),
                ("lodo_target_day", table["day_future"].to_numpy()),
            ]
        else:
            table, feat_cols = build_tray_pairs(variant)
            split_specs = [
                ("loo_trial", table["trial"].to_numpy()),
                ("lodo_target_day", table["day_future"].to_numpy()),
            ]

        out_csv = OUT_DIR / f"{level}_pairs_{variant}.csv"
        table.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv} ({len(table)} pairs, {len(feat_cols)} features)", flush=True)

        X = table[feat_cols].to_numpy(dtype=float)
        y = table["fw_future"].to_numpy(dtype=float)
        fw_t = table["fw_t"].to_numpy(dtype=float)

        for split_name, groups in split_specs:
            # Baselines
            for bname, bpred in (
                ("Persist", baseline_persist(fw_t)),
                ("GrowMean", baseline_grow_mean(fw_t, y, groups)),
            ):
                # GrowMean already LOO by group; Persist is constant
                if bname == "Persist":
                    pred = bpred
                else:
                    pred = bpred
                mae, rmse, r2 = metrics(y, pred)
                persist_mae, _, _ = metrics(y, baseline_persist(fw_t))
                results.append(
                    ForecastResult(
                        level=level,
                        variant=variant,
                        model=bname,
                        split=split_name,
                        n_rows=len(y),
                        n_features=0,
                        mae_g=mae,
                        rmse_g=rmse,
                        r2=r2,
                        mae_vs_persist=persist_mae - mae,
                    )
                )
                for i in range(len(y)):
                    pred_rows.append(
                        {
                            "level": level,
                            "variant": variant,
                            "model": bname,
                            "split": split_name,
                            "trial": table.iloc[i]["trial"],
                            "plant_id": int(table.iloc[i]["plant_id"]),
                            "day_t": int(table.iloc[i]["day_t"]),
                            "day_future": int(table.iloc[i]["day_future"]),
                            "fw_t": float(fw_t[i]),
                            "fw_future": float(y[i]),
                            "pred_fw_g": float(pred[i]),
                            "abs_err_g": float(abs(pred[i] - y[i])),
                        }
                    )
                print(
                    f"  [{level}/{variant}/{split_name}] {bname:12s} MAE={mae:.2f} g  R²={r2:.3f}",
                    flush=True,
                )

            persist_mae, _, _ = metrics(y, baseline_persist(fw_t))
            for name, model in build_regressors():
                pred = loo_groups(model, X, y, groups)
                mae, rmse, r2 = metrics(y, pred)
                results.append(
                    ForecastResult(
                        level=level,
                        variant=variant,
                        model=name,
                        split=split_name,
                        n_rows=len(y),
                        n_features=len(feat_cols),
                        mae_g=mae,
                        rmse_g=rmse,
                        r2=r2,
                        mae_vs_persist=persist_mae - mae,
                    )
                )
                for i in range(len(y)):
                    pred_rows.append(
                        {
                            "level": level,
                            "variant": variant,
                            "model": name,
                            "split": split_name,
                            "trial": table.iloc[i]["trial"],
                            "plant_id": int(table.iloc[i]["plant_id"]),
                            "day_t": int(table.iloc[i]["day_t"]),
                            "day_future": int(table.iloc[i]["day_future"]),
                            "fw_t": float(fw_t[i]),
                            "fw_future": float(y[i]),
                            "pred_fw_g": float(pred[i]),
                            "abs_err_g": float(abs(pred[i] - y[i])),
                        }
                    )
                beat = "beats persist" if mae < persist_mae else "vs persist"
                print(
                    f"  [{level}/{variant}/{split_name}] {name:12s} MAE={mae:.2f} g  "
                    f"R²={r2:.3f}  ({beat}: {persist_mae - mae:+.2f} g)",
                    flush=True,
                )

    return results, pred_rows


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="all", choices=["all", "plant", "tray"])
    parser.add_argument("--variants", default="compact,image,sensors")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    levels = ["plant", "tray"] if args.level == "all" else [args.level]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[ForecastResult] = []
    all_preds: list[dict] = []

    for level in levels:
        res, preds = run_level(level, variants)
        all_results.extend(res)
        all_preds.extend(preds)

    results_path = OUT_DIR / "results_next_checkpoint_forecast.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "level",
                "variant",
                "model",
                "split",
                "n_rows",
                "n_features",
                "mae_g",
                "rmse_g",
                "r2",
                "mae_vs_persist",
            ],
        )
        w.writeheader()
        for r in all_results:
            w.writerow(
                {
                    "level": r.level,
                    "variant": r.variant,
                    "model": r.model,
                    "split": r.split,
                    "n_rows": r.n_rows,
                    "n_features": r.n_features,
                    "mae_g": f"{r.mae_g:.4f}",
                    "rmse_g": f"{r.rmse_g:.4f}",
                    "r2": f"{r.r2:.6f}",
                    "mae_vs_persist": f"{r.mae_vs_persist:.4f}",
                }
            )
    print(f"\nWrote {results_path}", flush=True)

    pred_path = OUT_DIR / "predictions_next_checkpoint_forecast.csv"
    if all_preds:
        keys = list(all_preds[0].keys())
        with pred_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_preds)
        print(f"Wrote {pred_path}", flush=True)

    # Summary of best models
    summary = {"horizon_days": HORIZON, "best": {}}
    df = pd.DataFrame(
        [
            {
                "level": r.level,
                "variant": r.variant,
                "model": r.model,
                "split": r.split,
                "mae_g": r.mae_g,
                "r2": r.r2,
                "mae_vs_persist": r.mae_vs_persist,
                "n_rows": r.n_rows,
            }
            for r in all_results
        ]
    )
    for level in df["level"].unique():
        for split in df.loc[df.level == level, "split"].unique():
            sub = df[(df.level == level) & (df.split == split) & (~df.model.isin(["Persist"]))]
            if sub.empty:
                continue
            best = sub.sort_values("mae_g").iloc[0]
            persist = df[
                (df.level == level) & (df.split == split) & (df.model == "Persist") & (df.variant == best.variant)
            ]
            summary["best"][f"{level}/{split}"] = {
                "variant": best.variant,
                "model": best.model,
                "mae_g": float(best.mae_g),
                "r2": float(best.r2),
                "mae_vs_persist": float(best.mae_vs_persist),
                "persist_mae_g": float(persist.iloc[0].mae_g) if len(persist) else None,
                "n_rows": int(best.n_rows),
            }
    (OUT_DIR / "forecast_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_DIR / 'forecast_summary.json'}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()

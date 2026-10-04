#!/usr/bin/env python3
"""
Aim 1 Trial-2 check: train plant FW on Trial 1, predict per T2 cup,
aggregate mean/median, compare to observed tray median FW.

Writes under ml/plant/ only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROJECT = Path(__file__).resolve().parents[1]
PLANT = PROJECT / "ml" / "plant"
CUP = PROJECT / "segmentation" / "daily_cup_features.csv"
CHK2 = PROJECT / "ml" / "checkpoint_trial2.csv"
OBS = PROJECT / "Lettuce_model" / "data" / "observed_fresh_weight.csv"
OUT_DIR = PLANT

TARGET = "target_fw_g"
DROP = {
    "sample_id",
    "trial",
    "date",
    "window_start",
    "window_end",
    "label_source",
    "tray_median_fw_g",
    TARGET,
    "target_median_fw_g",
}

PLANT_IMG_COLS = [
    "plant_img_mask_area_px_wmean",
    "plant_img_mask_area_px_wstd",
    "plant_img_mask_area_px_last",
    "plant_img_mask_area_px_delta",
    "plant_img_plant_color_frac_wmean",
    "plant_img_plant_color_frac_last",
    "plant_img_blur_score_wmean",
]

VARIANTS = {
    "sensors": "plant_checkpoint_trial1_sensors.csv",
    "image": "plant_checkpoint_trial1_image.csv",
    "compact": "plant_checkpoint_trial1_compact.csv",
    "full": "plant_checkpoint_trial1.csv",
}


def plant_window_image_features(cups: pd.DataFrame, plant_id: int, w0: str, w1: str) -> dict:
    sub = cups.loc[
        (cups["cup_id"] == plant_id) & (cups["date"] >= w0) & (cups["date"] <= w1)
    ].sort_values("date")
    out = {c: np.nan for c in PLANT_IMG_COLS}
    out["n_plant_cup_days"] = int(sub["date"].nunique()) if len(sub) else 0
    if sub.empty:
        return out
    area = pd.to_numeric(sub["mask_area_px"], errors="coerce")
    color = pd.to_numeric(sub["plant_color_frac"], errors="coerce")
    blur = pd.to_numeric(sub["blur_score"], errors="coerce")
    out["plant_img_mask_area_px_wmean"] = float(area.mean()) if area.notna().any() else np.nan
    out["plant_img_mask_area_px_wstd"] = float(area.std()) if area.notna().sum() > 1 else np.nan
    out["plant_img_mask_area_px_last"] = float(area.iloc[-1]) if area.notna().any() else np.nan
    if area.notna().sum() >= 2:
        out["plant_img_mask_area_px_delta"] = float(area.iloc[-1] - area.iloc[0])
    out["plant_img_plant_color_frac_wmean"] = float(color.mean()) if color.notna().any() else np.nan
    out["plant_img_plant_color_frac_last"] = float(color.iloc[-1]) if color.notna().any() else np.nan
    out["plant_img_blur_score_wmean"] = float(blur.mean()) if blur.notna().any() else np.nan
    return out


def make_xgb():
    from xgboost import XGBRegressor

    return Pipeline(
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
                    reg_lambda=2.0,
                    random_state=42,
                    n_jobs=4,
                ),
            ),
        ]
    )


def load_train(path: Path):
    df = pd.read_csv(path)
    y = pd.to_numeric(df[TARGET], errors="coerce").to_numpy(dtype=float)
    Xdf = df.drop(columns=[c for c in DROP if c in df.columns], errors="ignore")
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
    return Xdf, y, list(Xdf.columns)


def build_t2_cup_rows(feat_cols: list[str]) -> pd.DataFrame:
    chk = pd.read_csv(CHK2)
    cups = pd.read_csv(CUP)
    cups = cups.loc[cups["trial"] == "trial2"].copy()
    cups["date"] = cups["date"].astype(str)
    cups["cup_id"] = pd.to_numeric(cups["cup_id"], errors="coerce")

    rows = []
    for _, r in chk.iterrows():
        day = int(r["day"])
        w0, w1 = str(r["window_start"]), str(r["window_end"])
        cup_ids = sorted(
            cups.loc[(cups["date"] >= w0) & (cups["date"] <= w1), "cup_id"].dropna().unique().astype(int)
        )
        base = r.to_dict()
        for cid in cup_ids:
            row = dict(base)
            row["trial"] = "trial2"
            row["plant_id"] = int(cid)
            row["cup_id"] = int(cid)
            row["sample_id"] = f"trial2_day{day}_cup{cid}"
            img = plant_window_image_features(cups, cid, w0, w1)
            row.update(img)
            row["has_plant_cup_image"] = int(row["n_plant_cup_days"] > 0)
            rows.append(row)

    df = pd.DataFrame(rows)
    # align columns to training features
    for c in feat_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(OBS)
    obs2 = obs.loc[obs["trial"] == 2, ["DAT", "observed_median_g"]].rename(
        columns={"DAT": "day", "observed_median_g": "tray_gt_median_g"}
    )

    cup_pred_rows = []
    tray_rows = []

    for fset, fname in VARIANTS.items():
        train_path = PLANT / fname
        Xtr, ytr, feat_cols = load_train(train_path)
        model = make_xgb()
        model.fit(Xtr.to_numpy(dtype=float), ytr)

        t2 = build_t2_cup_rows(feat_cols)
        Xte = t2[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        pred = model.predict(Xte).astype(float)
        t2 = t2.copy()
        t2["feature_set"] = fset
        t2["pred_fw_g"] = pred
        cup_pred_rows.append(
            t2[
                [
                    "feature_set",
                    "day",
                    "date",
                    "cup_id",
                    "pred_fw_g",
                    "has_plant_cup_image",
                    "n_plant_cup_days",
                ]
            ]
        )

        agg = (
            t2.groupby("day", as_index=False)
            .agg(
                n_cups=("cup_id", "nunique"),
                pred_mean_g=("pred_fw_g", "mean"),
                pred_median_g=("pred_fw_g", "median"),
            )
        )
        agg["feature_set"] = fset
        agg = agg.merge(obs2, on="day", how="left")
        # also checkpoint label (should match observed after phase0)
        chk = pd.read_csv(CHK2)[["day", "target_median_fw_g"]]
        agg = agg.merge(chk, on="day", how="left")
        agg["abs_err_mean_g"] = (agg["pred_mean_g"] - agg["tray_gt_median_g"]).abs()
        agg["abs_err_median_g"] = (agg["pred_median_g"] - agg["tray_gt_median_g"]).abs()
        tray_rows.append(agg)

        mae_mean = float(agg["abs_err_mean_g"].mean())
        mae_med = float(agg["abs_err_median_g"].mean())
        print(
            f"[{fset:8s}] T2 tray MAE vs GT: "
            f"pred_mean→{mae_mean:6.2f} g | pred_median→{mae_med:6.2f} g "
            f"(n_days={len(agg)})"
        )

    cups_out = pd.concat(cup_pred_rows, ignore_index=True)
    tray_out = pd.concat(tray_rows, ignore_index=True)
    cups_path = OUT_DIR / "t2_cup_fw_predictions.csv"
    tray_path = OUT_DIR / "t2_tray_aggregate_vs_gt.csv"
    cups_out.to_csv(cups_path, index=False)
    tray_out.to_csv(tray_path, index=False)

    summary = (
        tray_out.groupby("feature_set", as_index=False)
        .agg(
            mae_pred_mean_g=("abs_err_mean_g", "mean"),
            mae_pred_median_g=("abs_err_median_g", "mean"),
            n_days=("day", "nunique"),
        )
        .sort_values("mae_pred_median_g")
    )
    summary_path = OUT_DIR / "t2_tray_aggregate_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\n=== Summary (lower MAE better) ===")
    print(summary.to_string(index=False))
    print(f"\nWrote:\n  {cups_path}\n  {tray_path}\n  {summary_path}")


if __name__ == "__main__":
    main()

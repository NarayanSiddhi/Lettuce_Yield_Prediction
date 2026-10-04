"""
Build plant-level ML tables (10 weighed plants x 8 harvest days).

Writes ONLY under ml/plant/. Does not modify existing checkpoint_* files.

Each row is one Plant-ID on one harvest day (Day 0, 4, ..., 28).
Target is that plant's New Fresh Weight (g), not the tray median.

Usage:
  python3 export_plant_ml_datasets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "ml" / "plant"
TRACKER_PATH = PROJECT_ROOT / "Lettuce_FW_EC_Tracker_v3_2_.xlsx"
CHECKPOINT_PATH = PROJECT_ROOT / "master_checkpoint_dataset_trial1.csv"
CUP_FEATURES_PATH = PROJECT_ROOT / "segmentation" / "daily_cup_features.csv"

TARGET = "target_fw_g"
TRIAL = "trial1"
LABEL_SOURCE = "Lettuce_FW_EC_Tracker Data_Collection (per-plant)"

# Identifiers stored at the front of each table. day / plant_id are also model features.
ID_FRONT = [
    "sample_id",
    "trial",
    "day",
    "plant_id",
    "date",
    "window_start",
    "window_end",
    "window_days_count",
    "label_source",
]
# Never used as model inputs
META_ONLY = {
    "sample_id",
    "trial",
    "date",
    "window_start",
    "window_end",
    "label_source",
    "tray_median_fw_g",
    TARGET,
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


def load_plant_labels() -> pd.DataFrame:
    dc = pd.read_excel(TRACKER_PATH, sheet_name="Data_Collection")
    dc["day"] = pd.to_numeric(dc["Day"], errors="coerce")
    dc["plant_id"] = pd.to_numeric(dc["Plant-ID"], errors="coerce")
    dc["target_fw_g"] = pd.to_numeric(dc["New Fresh Weight (g)"], errors="coerce")
    labels = (
        dc.dropna(subset=["day", "plant_id", "target_fw_g"])
        .groupby(["day", "plant_id"], as_index=False)["target_fw_g"]
        .mean()
    )
    labels["day"] = labels["day"].astype(int)
    labels["plant_id"] = labels["plant_id"].astype(int)
    return labels


def plant_window_image_features(
    cups: pd.DataFrame, plant_id: int, window_start: str, window_end: str
) -> dict:
    sub = cups.loc[
        (cups["cup_id"] == plant_id)
        & (cups["date"] >= window_start)
        & (cups["date"] <= window_end)
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


def _unique(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def select_numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in META_ONLY:
            continue
        if c.endswith(("_wmin", "_wmax")):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            cols.append(c)
    return sorted(cols)


def write_xy(df: pd.DataFrame, feat_cols: list[str], stem: str) -> None:
    keep_ids = [c for c in ID_FRONT if c in df.columns]
    table = df[_unique(keep_ids + [TARGET] + feat_cols)].copy()
    table.to_csv(OUT_DIR / f"{stem}.csv", index=False)
    df[feat_cols].apply(pd.to_numeric, errors="coerce").to_csv(
        OUT_DIR / f"{stem}_X.csv", index=False
    )
    df[[TARGET]].to_csv(OUT_DIR / f"{stem}_y.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = load_plant_labels()
    checkpoint = pd.read_csv(CHECKPOINT_PATH)
    checkpoint["day"] = pd.to_numeric(checkpoint["day"], errors="coerce").astype(int)
    checkpoint["date"] = checkpoint["date"].astype(str)
    checkpoint["window_start"] = checkpoint["window_start"].astype(str)
    checkpoint["window_end"] = checkpoint["window_end"].astype(str)

    cups = pd.read_csv(CUP_FEATURES_PATH)
    cups = cups.loc[cups["trial"] == TRIAL].copy()
    cups["date"] = cups["date"].astype(str)
    cups["cup_id"] = pd.to_numeric(cups["cup_id"], errors="coerce")

    tray_median = checkpoint[["day", "target_median_fw_g"]].rename(
        columns={"target_median_fw_g": "tray_median_fw_g"}
    )
    day_meta = checkpoint.drop(columns=["target_median_fw_g", "sample_id"], errors="ignore")

    rows = []
    for _, lab in labels.iterrows():
        day = int(lab["day"])
        plant_id = int(lab["plant_id"])
        chk = day_meta.loc[day_meta["day"] == day]
        if chk.empty:
            continue
        row = chk.iloc[0].to_dict()
        row["trial"] = TRIAL
        row["plant_id"] = plant_id
        row["sample_id"] = f"{TRIAL}_day{day}_plant{plant_id}"
        row[TARGET] = float(lab["target_fw_g"])
        row["label_source"] = LABEL_SOURCE
        img = plant_window_image_features(
            cups, plant_id, str(row["window_start"]), str(row["window_end"])
        )
        row.update(img)
        row["has_plant_cup_image"] = int(row["n_plant_cup_days"] > 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.merge(tray_median, on="day", how="left")
    front = [c for c in ID_FRONT + [TARGET, "tray_median_fw_g"] if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest].sort_values(["day", "plant_id"]).reset_index(drop=True)

    full_feats = select_numeric_feature_cols(df)
    compact = sorted(
        {
            c
            for c in full_feats
            if c.endswith("_wmean")
            or c.startswith("win_img_")
            or c.startswith("plant_img_")
            or c in ("day", "plant_id", "window_days_count", "has_plant_cup_image", "n_plant_cup_days")
        }
    )
    image_only = sorted(
        c
        for c in full_feats
        if c.startswith("img_")
        or c.startswith("win_img_")
        or c.startswith("plant_img_")
        or c in ("day", "plant_id", "window_days_count", "has_plant_cup_image", "n_plant_cup_days")
    )
    sensors = sorted(
        c
        for c in full_feats
        if c.startswith("env_")
        or c.startswith("nutrient_")
        or c in ("day", "plant_id", "window_days_count")
    )

    write_xy(df, full_feats, "plant_checkpoint_trial1")
    write_xy(df, compact, "plant_checkpoint_trial1_compact")
    write_xy(df, image_only, "plant_checkpoint_trial1_image")
    write_xy(df, sensors, "plant_checkpoint_trial1_sensors")

    # Full table with metadata (tray median kept for comparison, not as a feature)
    df.to_csv(OUT_DIR / "plant_checkpoint_trial1_with_meta.csv", index=False)

    n_with_img = int(df["has_plant_cup_image"].sum())
    manifest = {
        "description": "Per-plant harvest-day tables. Does not replace ml/checkpoint_*.csv.",
        "trial": TRIAL,
        "n_rows": int(len(df)),
        "n_plants": int(df["plant_id"].nunique()),
        "n_harvest_days": int(df["day"].nunique()),
        "harvest_days": [int(x) for x in sorted(df["day"].unique())],
        "target_column": TARGET,
        "n_rows_with_matching_cup_image": n_with_img,
        "notes": [
            "Day 8: Plant-ID 6 appeared twice in the tracker; FW was averaged. Plant 10 has no Day-8 row.",
            "plant_img_* columns assume Plant-ID == camera cup_id; many harvest windows have no match.",
            "tray_median_fw_g is metadata only and is not in the X files.",
            "Existing ml/checkpoint_* files were not modified.",
        ],
        "files": {
            "full": "plant_checkpoint_trial1.csv",
            "full_X": "plant_checkpoint_trial1_X.csv",
            "full_y": "plant_checkpoint_trial1_y.csv",
            "compact": "plant_checkpoint_trial1_compact.csv",
            "image": "plant_checkpoint_trial1_image.csv",
            "sensors": "plant_checkpoint_trial1_sensors.csv",
            "with_meta": "plant_checkpoint_trial1_with_meta.csv",
        },
        "n_features": {
            "full": len(full_feats),
            "compact": len(compact),
            "image": len(image_only),
            "sensors": len(sensors),
        },
        "feature_columns_compact": compact,
        "feature_columns_image": image_only,
        "feature_columns_sensors": sensors,
        "feature_columns_full": full_feats,
    }
    (OUT_DIR / "feature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote plant-level datasets to {OUT_DIR}/")
    print(f"  rows: {len(df)}  plants: {df['plant_id'].nunique()}  harvest days: {sorted(df['day'].unique().tolist())}")
    print(f"  rows with a matching cup image in the 4-day window: {n_with_img}/{len(df)}")
    print(f"  features: full={len(full_feats)} compact={len(compact)} image={len(image_only)} sensors={len(sensors)}")
    print("  existing ml/checkpoint_* files were not modified")


if __name__ == "__main__":
    main()

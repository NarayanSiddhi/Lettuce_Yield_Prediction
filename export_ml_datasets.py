"""
Export ML-ready CSVs for tree models (XGBoost, Random Forest, etc.).

Reads master_checkpoint_dataset_*.csv and merged_daily_features_*.csv.

Outputs under ml/:
  checkpoint_{trial}.csv          — features + target + ids (modeling table)
  checkpoint_{trial}_X.csv        — numeric features only
  checkpoint_{trial}_y.csv          — target only
  checkpoint_combined.csv         — all trials (trial2 labels are protocol-shifted)
  daily_{trial}.csv               — per-day regression table (optional)
  feature_manifest.json           — column lists for train/serve
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
ML_DIR = PROJECT_ROOT / "ml"

META_COLS = {
    "sample_id",
    "trial",
    "day",
    "date",
    "target_median_fw_g",
    "window_start",
    "window_end",
    "label_source",
}
TARGET = "target_median_fw_g"
ID_COLS = ["sample_id", "trial", "day", "date", "window_start", "window_end", "window_days_count"]


def _is_feature_col(c: str) -> bool:
    if c in META_COLS:
        return False
    if c.endswith(("_wmin", "_wmax")):
        return False  # drop sparse extrema; keep wmean/wstd + win_* growth
    return True


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if not _is_feature_col(c):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
        else:
            try:
                pd.to_numeric(df[c])
                cols.append(c)
            except (TypeError, ValueError):
                pass
    # Drop all-NaN
    usable = []
    for c in cols:
        if df[c].notna().any():
            usable.append(c)
    return sorted(usable)


def feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "image_window": [],
        "image_growth": [],
        "sensors": [],
        "meta_numeric": [],
    }
    for c in feature_cols:
        if c.startswith("win_img_"):
            groups["image_growth"].append(c)
        elif c.startswith("img_") or c.startswith("win_img"):
            groups["image_window"].append(c)
        elif c.startswith("env_") or c.startswith("nutrient_"):
            groups["sensors"].append(c)
        elif c in ("day", "window_days_count", "exp_day"):
            groups["meta_numeric"].append(c)
        else:
            groups["meta_numeric"].append(c)
    return {k: v for k, v in groups.items() if v}


def export_checkpoint_trial(
    trial: str,
    label_source: str,
    out_dir: Path,
) -> dict:
    path = PROJECT_ROOT / f"master_checkpoint_dataset_{trial}.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df["label_source"] = label_source
    feat_cols = select_feature_columns(df)

    # Compact set for small-n tree models (image + growth + window meta)
    compact = [c for c in feat_cols if c.endswith("_wmean") or c.startswith("win_img_")]
    compact += [c for c in ("day", "window_days_count") if c in feat_cols]
    compact = sorted(set(compact))

    image_only = sorted(
        c
        for c in feat_cols
        if c.startswith("img_") or c.startswith("win_img_") or c in ("day", "window_days_count")
    )

    full_path = out_dir / f"checkpoint_{trial}.csv"
    df.to_csv(full_path, index=False)

    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[TARGET], errors="coerce")

    X.to_csv(out_dir / f"checkpoint_{trial}_X.csv", index=False)
    y.to_frame(name=TARGET).to_csv(out_dir / f"checkpoint_{trial}_y.csv", index=False)

    if compact:
        df[ID_COLS + [TARGET, "label_source"] + compact].to_csv(
            out_dir / f"checkpoint_{trial}_compact.csv", index=False
        )
    if image_only:
        df[ID_COLS + [TARGET, "label_source"] + image_only].to_csv(
            out_dir / f"checkpoint_{trial}_image.csv", index=False
        )

    return {
        "trial": trial,
        "n_rows": len(df),
        "n_features_full": len(feat_cols),
        "n_features_compact": len(compact),
        "n_features_image_only": len(image_only),
        "target": TARGET,
        "label_source": label_source,
        "id_columns": [c for c in ID_COLS if c in df.columns],
        "feature_columns": feat_cols,
        "feature_columns_compact": compact,
        "feature_columns_image_only": image_only,
        "feature_groups": feature_groups(feat_cols),
        "files": {
            "full": f"checkpoint_{trial}.csv",
            "X": f"checkpoint_{trial}_X.csv",
            "y": f"checkpoint_{trial}_y.csv",
            "compact": f"checkpoint_{trial}_compact.csv",
            "image": f"checkpoint_{trial}_image.csv",
        },
    }


def export_daily_trial(trial: str, out_dir: Path) -> None:
    path = PROJECT_ROOT / f"merged_daily_features_{trial}.csv"
    if not path.exists():
        return
    daily = pd.read_csv(path)
    labels = pd.read_csv(PROJECT_ROOT / f"master_checkpoint_dataset_{trial}.csv")[
        ["date", TARGET, "day"]
    ]
    daily = daily.merge(labels, on="date", how="left", suffixes=("", "_chk"))
    daily["has_checkpoint_label"] = daily[TARGET].notna()
    daily.to_csv(out_dir / f"daily_{trial}.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=ML_DIR)
    args = p.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "description": "Lettuce yield ML tables from timelapse cup segmentation + sensors",
        "target_column": TARGET,
        "trials": {},
        "notes": [
            "checkpoint_trial1: primary dataset (8 harvest checkpoints, trial-1 FW labels).",
            "checkpoint_trial2: same Day 0-28 label curve shifted to trial-2 start; verify before use.",
            "Use checkpoint_trial1_image.csv (image+growth only) for n=8 tree models.",
            "Use *_compact.csv for full sensor+image window means.",
            "Drop rows with window_days_count < 2 for stricter training.",
        ],
    }

    t1 = export_checkpoint_trial("trial1", "Lettuce_FW_EC_Tracker Data_Collection", out_dir)
    t2 = export_checkpoint_trial(
        "trial2",
        "trial1_protocol_shifted_start_2026-01-20",
        out_dir,
    )
    manifest["trials"]["trial1"] = t1
    manifest["trials"]["trial2"] = t2

    c1 = pd.read_csv(out_dir / "checkpoint_trial1.csv")
    c2 = pd.read_csv(out_dir / "checkpoint_trial2.csv")
    combined = pd.concat([c1, c2], ignore_index=True)
    combined.to_csv(out_dir / "checkpoint_combined.csv", index=False)

    feat = t1["feature_columns"]
    pd.concat(
        [
            c1[ID_COLS + [TARGET] + feat],
            c2[ID_COLS + [TARGET] + feat],
        ],
        ignore_index=True,
    ).to_csv(out_dir / "checkpoint_combined_Xy.csv", index=False)

    export_daily_trial("trial1", out_dir)
    export_daily_trial("trial2", out_dir)

    manifest_path = out_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote ML datasets to {out_dir}/")
    print(f"  trial1: {t1['n_rows']} rows, {t1['n_features_full']} features ({t1['n_features_compact']} compact)")
    print(f"  trial2: {t2['n_rows']} rows, {t2['n_features_full']} features")
    print(f"  manifest: {manifest_path.name}")
    print("\nPrimary files for XGBoost / RF:")
    print("  ml/checkpoint_trial1_image.csv   (recommended, ~30 image features)")
    print("  ml/checkpoint_trial1_compact.csv (image + sensors)")


if __name__ == "__main__":
    main()

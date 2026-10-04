"""
Merge cup-level daily image features with sensors + FW checkpoints.

Enhancements for ML:
  - Full timelapse calendar per trial (sensor rows even when image missing)
  - Tray coverage fraction, experimental day, daily growth deltas
  - Checkpoint windows with growth/slope features

Outputs (repo root, per trial):
  image_features_daily_{trial}.csv
  merged_daily_features_{trial}.csv
  master_checkpoint_dataset_{trial}.csv
  coverage_report_{trial}.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_master_dataset_trial1 import (
    build_checkpoint_dataset,
    build_sensor_daily_features,
    load_fw_labels_by_day,
)

CUP_FEATURES_PATH = Path("segmentation/daily_cup_features.csv")
TIMELAPSE_INDEX_PATH = Path("segmentation/timelapse_index.csv")
CUPS_PER_TRAY = 14
NUMERIC_CUP_COLS = ("mask_area_px", "plant_color_frac", "blur_score")

# Daily image cols used for window growth features at checkpoints
GROWTH_TRACK_COLS = (
    "img_mask_area_sum",
    "img_mask_area_mean",
    "img_plant_color_mean",
    "img_n_cups",
    "img_coverage_frac",
)


def load_timelapse_calendar(project_root: Path) -> pd.DataFrame:
    idx = pd.read_csv(project_root / TIMELAPSE_INDEX_PATH)
    cal = (
        idx.groupby(["trial", "capture_date"], as_index=False)
        .size()
        .rename(columns={"capture_date": "date", "size": "timelapse_frames"})
    )
    cal["date"] = cal["date"].astype(str)
    return cal


def aggregate_tray_daily(cup_df: pd.DataFrame, trial: str) -> pd.DataFrame:
    sub = cup_df.loc[cup_df["trial"] == trial].copy()
    for col in NUMERIC_CUP_COLS:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")

    agg = (
        sub.groupby("date", as_index=False)
        .agg(
            img_n_cups=("cup_id", "nunique"),
            img_mask_area_mean=("mask_area_px", "mean"),
            img_mask_area_sum=("mask_area_px", "sum"),
            img_mask_area_max=("mask_area_px", "max"),
            img_mask_area_std=("mask_area_px", "std"),
            img_plant_color_mean=("plant_color_frac", "mean"),
            img_plant_color_std=("plant_color_frac", "std"),
            img_plant_color_min=("plant_color_frac", "min"),
            img_plant_color_max=("plant_color_frac", "max"),
            img_blur_mean=("blur_score", "mean"),
            img_blur_min=("blur_score", "min"),
            img_blur_max=("blur_score", "max"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    agg["img_coverage_frac"] = agg["img_n_cups"] / float(CUPS_PER_TRAY)
    agg.insert(0, "trial", trial)
    return agg


def enrich_daily_timeseries(daily: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    """Experimental day + day-over-day growth on key image metrics."""
    out = daily.sort_values("date").copy()
    out["date"] = pd.to_datetime(out["date"])
    out["exp_day"] = (out["date"] - start_date).dt.days.astype("Int64")

    for col in GROWTH_TRACK_COLS:
        if col not in out.columns:
            continue
        out[f"{col}_d1"] = out[col].diff()
        out[f"{col}_pct1"] = out[col].pct_change()

    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def merge_calendar_tray_sensors(
    calendar: pd.DataFrame,
    tray: pd.DataFrame,
    sensor_daily: pd.DataFrame,
    trial: str,
) -> pd.DataFrame:
    """One row per timelapse calendar date; image may be NaN, sensors filled."""
    cal = calendar.loc[calendar["trial"] == trial, ["date", "timelapse_frames"]].copy()
    img_cols = [c for c in tray.columns if c not in ("trial", "date")]
    merged = cal.merge(tray.drop(columns=["trial"], errors="ignore"), on="date", how="left")
    merged = merged.merge(sensor_daily, on="date", how="left")
    merged.insert(0, "trial", trial)
    return merged


def _window_slope(values: np.ndarray, days: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(days)
    if mask.sum() < 2:
        return float("nan")
    x = days[mask].astype(float)
    y = values[mask].astype(float)
    if np.std(x) == 0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def add_checkpoint_growth_features(
    checkpoint: pd.DataFrame,
    merged_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Per-checkpoint slopes/deltas for image metrics over the harvest window."""
    md = merged_daily.copy()
    md["date"] = pd.to_datetime(md["date"], errors="coerce")
    out = checkpoint.copy()

    for _, r in out.iterrows():
        w_start = pd.to_datetime(r["window_start"])
        w_end = pd.to_datetime(r["window_end"])
        window = md[(md["date"] >= w_start) & (md["date"] <= w_end)].sort_values("date")
        if window.empty:
            continue
        exp_days = window["exp_day"].to_numpy(dtype=float) if "exp_day" in window.columns else np.arange(len(window), dtype=float)

        for col in GROWTH_TRACK_COLS:
            if col not in window.columns:
                continue
            s = pd.to_numeric(window[col], errors="coerce")
            vals = s.to_numpy(dtype=float)
            valid = np.isfinite(vals)
            prefix = f"win_{col}"
            out.loc[out["day"] == r["day"], f"{prefix}_slope"] = _window_slope(vals, exp_days)
            if valid.sum() >= 2:
                first = vals[valid][0]
                last = vals[valid][-1]
                out.loc[out["day"] == r["day"], f"{prefix}_delta"] = float(last - first)
                out.loc[out["day"] == r["day"], f"{prefix}_last"] = float(last)
                if np.isfinite(first) and first != 0:
                    out.loc[out["day"] == r["day"], f"{prefix}_pct_delta"] = float((last - first) / first)
            elif valid.sum() == 1:
                out.loc[out["day"] == r["day"], f"{prefix}_last"] = float(vals[valid][0])

    return out


def write_coverage_report(
    path: Path,
    trial: str,
    calendar: pd.DataFrame,
    tray: pd.DataFrame,
    merged: pd.DataFrame,
    checkpoint: pd.DataFrame,
) -> None:
    cal_dates = set(calendar.loc[calendar["trial"] == trial, "date"])
    tray_dates = set(tray.loc[tray["trial"] == trial, "date"])
    has_img = merged.loc[merged["trial"] == trial, "img_n_cups"].notna() & (merged["img_n_cups"] > 0)
    img_dates = set(merged.loc[merged["trial"] == trial].loc[has_img, "date"])

    lines = [
        f"Coverage report — {trial}",
        f"  Timelapse calendar dates: {len(cal_dates)}",
        f"  Dates with cup features:  {len(tray_dates)}",
        f"  Dates with img_n_cups>0:  {len(img_dates)}",
        f"  Merged daily rows:        {len(merged[merged['trial'] == trial])}",
        "",
        "Checkpoints (window_days_count):",
    ]
    for _, r in checkpoint.iterrows():
        lines.append(
            f"  day {int(r['day']):2d}  {r['date']}  "
            f"window={int(r['window_days_count'])}  "
            f"target={float(r['target_median_fw_g']):.2f} g"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_trial_outputs(
    project_root: Path,
    trial: str,
    cup_df: pd.DataFrame,
    sensor_daily: pd.DataFrame,
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tray = aggregate_tray_daily(cup_df, trial)
    start = pd.to_datetime(tray["date"].min())

    merged = merge_calendar_tray_sensors(calendar, tray, sensor_daily, trial)
    merged = enrich_daily_timeseries(merged.drop(columns=["trial"], errors="ignore"), start)
    merged.insert(0, "trial", trial)

    image_path = project_root / f"image_features_daily_{trial}.csv"
    merged_path = project_root / f"merged_daily_features_{trial}.csv"
    checkpoint_path = project_root / f"master_checkpoint_dataset_{trial}.csv"
    report_path = project_root / f"coverage_report_{trial}.txt"

    tray.to_csv(image_path, index=False)
    merged.to_csv(merged_path, index=False)

    # Checkpoint windows (numeric stats); use trial-only daily without duplicate trial col
    chk_input = merged.drop(columns=["trial", "timelapse_frames"], errors="ignore")
    checkpoint = build_checkpoint_dataset(project_root, chk_input, start_date=start)
    checkpoint = add_checkpoint_growth_features(checkpoint, merged)
    checkpoint.insert(0, "trial", trial)
    checkpoint.insert(1, "sample_id", [f"{trial}_day{int(d)}" for d in checkpoint["day"]])
    checkpoint.to_csv(checkpoint_path, index=False)

    write_coverage_report(report_path, trial, calendar, tray, merged, checkpoint)

    print(f"\n=== {trial} ===")
    print(f"  calendar dates: {len(calendar[calendar['trial'] == trial])}")
    print(f"  cup-feature dates: {len(tray)}  ({tray['date'].min()} .. {tray['date'].max()})")
    print(f"  start_date (Day 0): {start.strftime('%Y-%m-%d')}")
    print(f"  saved: {image_path.name}, {merged_path.name}, {checkpoint_path.name}")
    print(f"  coverage: {report_path.name}")
    wdc = checkpoint["window_days_count"]
    print(f"  checkpoint windows: min={int(wdc.min())} max={int(wdc.max())} mean={wdc.mean():.1f}")

    return merged, checkpoint


def main() -> None:
    project_root = Path(__file__).resolve().parent
    cup_path = project_root / CUP_FEATURES_PATH
    if not cup_path.exists():
        raise SystemExit(f"Missing {cup_path}. Run: bash segmentation/run_post_segmentation.sh")

    cup_df = pd.read_csv(cup_path)
    cup_df["date"] = cup_df["date"].astype(str)
    calendar = load_timelapse_calendar(project_root)

    print("Building daily sensor features from Ambient_nutrient_data.xlsx ...")
    sensor_daily = build_sensor_daily_features(project_root)
    sensor_daily.to_csv(project_root / "sensor_features_daily.csv", index=False)

    trials = sorted(cup_df["trial"].dropna().unique())
    for trial in trials:
        build_trial_outputs(project_root, trial, cup_df, sensor_daily, calendar)

    print("\nNext: python3 export_ml_datasets.py")


if __name__ == "__main__":
    main()

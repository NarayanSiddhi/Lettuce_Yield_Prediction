#!/usr/bin/env python3
"""
Phase 0 data fixes (dataset audit action items):
1) Rebuild sensor_features_daily_trial2.csv from Trial2 Mycodo (replace T1 duplicate).
2) Patch Trial 2 fresh-weight labels in ml/checkpoint_*.csv from observed_fresh_weight.csv.
3) Write research master timeline + canonical tray labels.
4) Archive the bogus T2 sensor file.

Run time: usually < 2 minutes (no GPU).
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
RESEARCH = PROJECT / "research"
ARCHIVE = RESEARCH / "archive_phase0_20260914"
OUT_TABLES = RESEARCH / "tables"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def archive_bad_t2_sensor() -> None:
    src = PROJECT / "sensor_features_daily_trial2.csv"
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return
    dst = ARCHIVE / "sensor_features_daily_trial2.INVALID_t1_copy.csv"
    if not dst.exists():
        shutil.copy2(src, dst)
        print(f"Archived bogus T2 sensors → {dst} (md5={md5(src)})")


def build_trial2_sensor_daily() -> pd.DataFrame:
    """Day/night aggregates from Mycodo. No actuators available for Trial 2."""
    path = PROJECT / "Data collection" / "Trial2_Mycodo_raw_export.xlsx"
    df = pd.read_excel(path)
    rename = {
        "DateTime": "datetime",
        "Atlas pH (CH0, Ion Concentration, pH)": "ph",
        "Atlas CO2 (Carbon Dioxide Gas) (CH0, Carbon Dioxide, ppm)": "co2_ppm",
        "Atlas EC (CH0, Electrical Conductivity, μS/cm)": "ec_us_cm",
        "BME280 (CH0, Temperature, °C)": "air_temp_c",
        "BME280 (CH1, Humidity, %)": "humidity_pct",
        "BME280 (CH5, Vapor Pressure Deficit, Pa)": "vpd_pa",
    }
    df = df.rename(columns=rename)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()
    df["date"] = df["datetime"].dt.date.astype(str)
    df["hour"] = df["datetime"].dt.hour
    # Match camera photoperiod window used in this project (~05:00–20:59 lights)
    df["period"] = np.where((df["hour"] >= 5) & (df["hour"] < 21), "day", "night")

    df["co2_ppm_raw"] = df["co2_ppm"]
    df["co2_ppm"] = df["co2_ppm"].where(df["co2_ppm"] >= 300)

    day_rows, night_rows = [], []
    for d, g in df.groupby("date"):
        for period, bucket in (("day", day_rows), ("night", night_rows)):
            gp = g[g["period"] == period]
            row: dict = {"date": d}
            pref = f"env_{period}"
            for c in ["co2_ppm", "air_temp_c", "humidity_pct", "vpd_pa"]:
                s = pd.to_numeric(gp[c], errors="coerce")
                row[f"{pref}_{c}_mean"] = float(s.mean()) if s.notna().any() else np.nan
                row[f"{pref}_{c}_std"] = float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan
                row[f"{pref}_{c}_min"] = float(s.min()) if s.notna().any() else np.nan
                row[f"{pref}_{c}_max"] = float(s.max()) if s.notna().any() else np.nan
            bucket.append(row)
    day_df = pd.DataFrame(day_rows)
    night_df = pd.DataFrame(night_rows)

    nut_rows = []
    for d, g in df.groupby("date"):
        row = {"date": d}
        for c, name in [("ph", "ph"), ("ec_us_cm", "ec_us_cm")]:
            s = pd.to_numeric(g[c], errors="coerce")
            row[f"nutrient_{name}_mean"] = float(s.mean()) if s.notna().any() else np.nan
            row[f"nutrient_{name}_std"] = float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan
            row[f"nutrient_{name}_min"] = float(s.min()) if s.notna().any() else np.nan
            row[f"nutrient_{name}_max"] = float(s.max()) if s.notna().any() else np.nan
        ec_mean = row.get("nutrient_ec_us_cm_mean", np.nan)
        row["nutrient_ec_ms_cm_mean"] = (ec_mean / 1000.0) if np.isfinite(ec_mean) else np.nan
        row["n_samples"] = int(len(g))
        row["n_co2_valid"] = int(pd.to_numeric(g["co2_ppm"], errors="coerce").notna().sum())
        row["n_co2_raw_lt_300"] = int((pd.to_numeric(g["co2_ppm_raw"], errors="coerce") < 300).sum())
        nut_rows.append(row)
    nut_df = pd.DataFrame(nut_rows)

    daily = night_df.merge(day_df, on="date", how="outer").merge(nut_df, on="date", how="outer")
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["has_actuators"] = False
    daily["trial"] = "trial2"
    return daily


def canonical_fw() -> pd.DataFrame:
    fw = pd.read_csv(PROJECT / "Lettuce_model" / "data" / "observed_fresh_weight.csv")
    # Map DAT → calendar date using protocol starts
    starts = {"1": date(2025, 10, 14), "2": date(2026, 1, 19)}
    # Also support int trial
    rows = []
    for _, r in fw.iterrows():
        trial_num = str(int(r["trial"]))
        trial = f"trial{trial_num}"
        dat = int(r["DAT"])
        d = starts[trial_num] + timedelta(days=dat)
        rows.append(
            {
                "trial": trial,
                "DAT": dat,
                "date": d.isoformat(),
                "observed_median_fw_g": float(r["observed_median_g"]),
                "target_low_g": float(r["target_low_g"]),
                "target_mid_g": float(r["target_mid_g"]),
                "target_high_g": float(r["target_high_g"]),
                "label_source": "Lettuce_model/data/observed_fresh_weight.csv",
            }
        )
    return pd.DataFrame(rows)


def patch_checkpoint_labels(fw: pd.DataFrame) -> None:
    """Replace Trial2 target_median_fw_g with canonical observed medians (by DAT)."""
    fw2 = fw[fw["trial"] == "trial2"].set_index("DAT")["observed_median_fw_g"].to_dict()
    files = [
        PROJECT / "ml" / "checkpoint_trial2.csv",
        PROJECT / "ml" / "checkpoint_trial2_compact.csv",
        PROJECT / "ml" / "checkpoint_trial2_image.csv",
        PROJECT / "ml" / "checkpoint_combined.csv",
    ]
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for path in files:
        if not path.exists():
            continue
        shutil.copy2(path, ARCHIVE / f"{path.name}.pre_label_fix")
        df = pd.read_csv(path)
        if "day" not in df.columns or "target_median_fw_g" not in df.columns:
            continue
        mask = df["trial"].astype(str).str.lower().isin(["trial2", "2"])
        before = df.loc[mask, "target_median_fw_g"].tolist()
        df.loc[mask, "target_median_fw_g"] = df.loc[mask, "day"].map(lambda d: fw2.get(int(d), np.nan))
        if "label_source" in df.columns:
            df.loc[mask, "label_source"] = "observed_fresh_weight.csv (phase0 fix 2026-09-14)"
        # also fix y vector if companion exists
        df.to_csv(path, index=False)
        after = df.loc[mask, "target_median_fw_g"].tolist()
        print(f"Patched {path.name}: {before} → {after}")

    y_path = PROJECT / "ml" / "checkpoint_trial2_y.csv"
    if y_path.exists():
        shutil.copy2(y_path, ARCHIVE / "checkpoint_trial2_y.csv.pre_label_fix")
        # rebuild from patched trial2
        t2 = pd.read_csv(PROJECT / "ml" / "checkpoint_trial2.csv")
        t2[["sample_id", "target_median_fw_g"]].to_csv(y_path, index=False)
        print(f"Patched {y_path.name}")


def build_master_timeline(sensor_t2: pd.DataFrame, fw: pd.DataFrame) -> pd.DataFrame:
    idx = pd.read_csv(PROJECT / "segmentation" / "timelapse_index.csv")
    idx["capture_date"] = idx["capture_date"].astype(str)
    img = (
        idx.groupby(["trial", "capture_date"], as_index=False)
        .agg(n_images=("source_path", "count"), mean_brightness=("mean_brightness", "mean"))
        .rename(columns={"capture_date": "date"})
    )

    starts = {"trial1": date(2025, 10, 14), "trial2": date(2026, 1, 19)}
    ends = {"trial1": date(2025, 11, 11), "trial2": date(2026, 2, 16)}

    sensor_t1 = pd.read_csv(PROJECT / "sensor_features_daily_trial1.csv")
    sensor_t1["trial"] = "trial1"
    # keep compact sensor presence flags
    t1_keep = ["date", "trial"] + [c for c in sensor_t1.columns if c.endswith("_mean") and any(
        k in c for k in ("co2", "air_temp", "humidity", "vpd", "nutrient_ph", "nutrient_ec")
    )][:20]
    t1_small = sensor_t1[[c for c in t1_keep if c in sensor_t1.columns]].copy()

    t2_keep = ["date", "trial"] + [c for c in sensor_t2.columns if c.endswith("_mean")]
    t2_small = sensor_t2[t2_keep].copy()

    rows = []
    for trial, start, end in (
        ("trial1", starts["trial1"], ends["trial1"]),
        ("trial2", starts["trial2"], ends["trial2"]),
    ):
        d = start
        while d <= end:
            rows.append({"trial": trial, "date": d.isoformat(), "DAT": (d - start).days})
            d += timedelta(days=1)
    timeline = pd.DataFrame(rows)
    timeline = timeline.merge(img, on=["trial", "date"], how="left")
    timeline["n_images"] = timeline["n_images"].fillna(0).astype(int)
    timeline = timeline.merge(fw.drop(columns=[c for c in fw.columns if c.startswith("target_")]), on=["trial", "date", "DAT"], how="left")
    timeline = timeline.rename(columns={"observed_median_fw_g": "fw_median_g"})
    timeline["is_fw_checkpoint"] = timeline["fw_median_g"].notna()

    # merge key sensors
    sens = pd.concat([t1_small, t2_small], ignore_index=True, sort=False)
    timeline = timeline.merge(sens, on=["trial", "date"], how="left")
    timeline["has_sensor_row"] = timeline["date"].isin(set(sens["date"])) & (
        ((timeline["trial"] == "trial1") & timeline["date"].isin(set(t1_small["date"])))
        | ((timeline["trial"] == "trial2") & timeline["date"].isin(set(t2_small["date"])))
    )
    # simpler has_sensor
    t1_dates = set(t1_small["date"])
    t2_dates = set(t2_small["date"])
    timeline["has_sensor_row"] = timeline.apply(
        lambda r: (r["date"] in t1_dates) if r["trial"] == "trial1" else (r["date"] in t2_dates),
        axis=1,
    )

    # t→t+4 feasibility
    date_sets = {t: set(timeline.loc[timeline["trial"] == t, "date"]) for t in ("trial1", "trial2")}
    img_dates = {
        t: set(timeline.loc[(timeline["trial"] == t) & (timeline["n_images"] > 0), "date"])
        for t in ("trial1", "trial2")
    }

    def has_pair(r):
        d0 = date.fromisoformat(r["date"])
        d4 = (d0 + timedelta(days=4)).isoformat()
        return int(r["date"] in img_dates[r["trial"]] and d4 in img_dates[r["trial"]])

    timeline["has_image_t_and_t4"] = timeline.apply(has_pair, axis=1)
    return timeline.sort_values(["trial", "DAT"]).reset_index(drop=True)


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    archive_bad_t2_sensor()

    print("Building Trial 2 daily sensors from Mycodo...")
    sensor_t2 = build_trial2_sensor_daily()
    out_sensor = PROJECT / "sensor_features_daily_trial2.csv"
    sensor_t2.to_csv(out_sensor, index=False)
    sensor_t2.to_csv(OUT_TABLES / "sensor_features_daily_trial2.csv", index=False)
    print(f"Wrote {out_sensor} ({len(sensor_t2)} days, {sensor_t2.shape[1]} cols)")
    print("  date range:", sensor_t2["date"].iloc[0], "→", sensor_t2["date"].iloc[-1])
    # prove not identical to T1
    t1 = PROJECT / "sensor_features_daily_trial1.csv"
    print("  identical_to_t1?", md5(out_sensor) == md5(t1))

    fw = canonical_fw()
    fw.to_csv(OUT_TABLES / "fw_tray_canonical.csv", index=False)
    print(f"Wrote {OUT_TABLES / 'fw_tray_canonical.csv'}")

    patch_checkpoint_labels(fw)

    timeline = build_master_timeline(sensor_t2, fw)
    tl_path = OUT_TABLES / "master_timeline.csv"
    timeline.to_csv(tl_path, index=False)
    print(f"Wrote {tl_path} ({len(timeline)} rows)")
    for trial in ("trial1", "trial2"):
        sub = timeline[timeline["trial"] == trial]
        print(
            f"  {trial}: days={len(sub)} images_days={(sub.n_images>0).sum()} "
            f"fw_checkpoints={sub.is_fw_checkpoint.sum()} "
            f"sensor_days={sub.has_sensor_row.sum()} "
            f"t_t4_image_pair_starts={sub.has_image_t_and_t4.sum()}"
        )
    print("Phase 0 data fix done.")


if __name__ == "__main__":
    main()

"""Build daily driver tables for the two grow-tent lettuce trials.

Reads the raw Mycodo exports that came out of the Raspberry Pi controller and
collapses them to one row per day-after-transplant (DAT), which is the time
step the crop model integrates on.

Sources -- all under `Data collection/`
--------------------------------------
Trial 1 (adaptive EC, 2025-10-14 -> 2025-11-11)
    Trial1_environment_log.csv   air T, RH, CO2, pressure + actuator durations
    Trial1_nutrient_log.csv      solution EC, solution T, reservoir volume

Trial 2 (fixed EC, 2026-01-19 -> 2026-02-16)
    Trial2_Mycodo_raw_export.xlsx    full-resolution Mycodo export

Fresh weight, both trials
    FreshWeight_checkpoints_both_trials.xlsx

Trial 1 per-plant weights and the operator's EC decisions
    Trial1_FreshWeight_EC_tracker.xlsx

PPFD is NOT logged by either trial -- there was no PAR sensor in the loop, only
manual fixture dimming.  It is therefore reconstructed from the documented
light schedule in the thesis (Table 3.3) and written out explicitly so the
assumption stays visible.

Usage
-----
    python prepare_drivers.py                 # writes ../data/*_daily_drivers.csv
    python prepare_drivers.py --check         # print summaries, write nothing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent                      # project root (sibling of Data collection/)
DATA_OUT = HERE.parent / "data"

RAW = PROJECT / "Data collection"

TRIAL1_ENV = RAW / "Trial1_environment_log.csv"
TRIAL1_NUTR = RAW / "Trial1_nutrient_log.csv"
TRIAL2_XLSX = RAW / "Trial2_Mycodo_raw_export.xlsx"
FW_XLSX = RAW / "FreshWeight_checkpoints_both_trials.xlsx"
TRACKER_XLSX = RAW / "Trial1_FreshWeight_EC_tracker.xlsx"

# --------------------------------------------------------------------------
# Trial definitions (thesis Table 3.3 + camera-capture date ranges)
# --------------------------------------------------------------------------
# Transplant dates are pinned by the first logger record of each trial, which
# in both cases is an evening timestamp on the day the seedlings went in:
#   Trial 1  first camera frame  2025-10-14 19:20:54
#   Trial 2  first Mycodo record 2026-01-19 19:20:36
# Adding the 28-day cycle then lands harvest exactly on the last day of data
# in both trials (2025-11-11 and 2026-02-16), which is the consistency check
# that settles the DAT numbering.  Note the two camera folders are offset by
# one day as a result: Trial 1 images run DAT 0-27, Trial 2 images DAT 1-28.
TRIALS = {
    1: dict(
        label="Trial 1 - adaptive EC steering",
        transplant="2025-10-14",
        harvest="2025-11-11",
        photoperiod_h=16.0,
        # PPFD [umol m-2 s-1] as (start_DAT, value).  Step change on DAT 12.
        ppfd_schedule=[(0, 200.0), (12, 445.0)],
    ),
    2: dict(
        label="Trial 2 - fixed EC (1.4-1.6 mS/cm)",
        transplant="2026-01-19",
        harvest="2026-02-16",
        photoperiod_h=16.0,
        ppfd_schedule=[(0, 445.0)],
    ),
}

LIGHT_ON_HOUR = 5     # 05:00 America/Los_Angeles
LIGHT_OFF_HOUR = 21   # 21:00

# Plausibility windows applied before averaging.  The Atlas CO2 NDIR circuit in
# particular emits sub-ppm values during warm-up and after I2C read errors --
# 1358 of 5549 Trial 2 readings are below 300 ppm, which would drag the daily
# mean down by ~120 ppm if averaged in.  Anything outside these ranges is
# dropped as an instrument artefact, not carried as data.
QC_RANGES = {
    "Temperature": (5.0, 45.0),          # air, degC
    "Humidity": (10.0, 100.0),           # %
    "Carbon Dioxide": (300.0, 2000.0),   # ppm
    "EC_uS": (200.0, 5000.0),            # uS/cm
    "Electrical Conductivity": (200.0, 5000.0),
    "pH": (3.0, 10.0),
    "Pressure": (80000.0, 110000.0),     # Pa
}


def ppfd_for_dat(dat: int, schedule) -> float:
    """PPFD on a given day after transplant from a step schedule."""
    value = schedule[0][1]
    for start, val in schedule:
        if dat >= start:
            value = val
    return value


def _daily_from_timeseries(df, time_col, cols, transplant, harvest):
    """Collapse an irregular time series to 24-h and photoperiod daily means."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    t0 = pd.Timestamp(transplant)
    t1 = pd.Timestamp(harvest) + pd.Timedelta(days=1)
    df = df[(df[time_col] >= t0) & (df[time_col] < t1)]

    for col in cols:
        lo, hi = QC_RANGES.get(col, (-np.inf, np.inf))
        present = df[col].notna()
        bad = present & ~df[col].between(lo, hi)
        n_bad = int(bad.sum())
        if n_bad:
            print(f"    QC: dropped {n_bad} of {int(present.sum())} "
                  f"'{col}' readings outside [{lo}, {hi}]")
        df.loc[bad, col] = np.nan

    df["DAT"] = (df[time_col] - t0).dt.days
    in_light = df[time_col].dt.hour.between(LIGHT_ON_HOUR, LIGHT_OFF_HOUR - 1)

    full = df.groupby("DAT")[cols].mean()
    light = df[in_light].groupby("DAT")[cols].mean()
    light.columns = [f"{c}__light" for c in light.columns]
    return full.join(light, how="left")


def build_trial1() -> pd.DataFrame:
    cfg = TRIALS[1]
    env = pd.read_csv(TRIAL1_ENV)
    nutr = pd.read_csv(TRIAL1_NUTR)

    env_daily = _daily_from_timeseries(
        env, "DateTime", ["Temperature", "Humidity", "Carbon Dioxide", "Pressure"],
        cfg["transplant"], cfg["harvest"])
    nutr_daily = _daily_from_timeseries(
        nutr, "DateTime", ["Electrical Conductivity", "Temperature"],
        cfg["transplant"], cfg["harvest"])

    out = pd.DataFrame(index=env_daily.index)
    out["T_air_C"] = env_daily["Temperature"]
    out["T_air_light_C"] = env_daily["Temperature__light"]
    out["RH_pct"] = env_daily["Humidity"]
    out["RH_light_pct"] = env_daily["Humidity__light"]
    out["CO2_ppm"] = env_daily["Carbon Dioxide"]
    out["P_air_Pa"] = env_daily["Pressure"]
    out["EC_mS_cm"] = nutr_daily["Electrical Conductivity"] / 1000.0
    out["T_solution_C"] = nutr_daily["Temperature"]
    return _finish(out, cfg)


def build_trial2() -> pd.DataFrame:
    cfg = TRIALS[2]
    raw = pd.read_excel(TRIAL2_XLSX)
    rename = {}
    for c in raw.columns:
        lc = str(c).lower()
        if lc.startswith("datetime"):
            rename[c] = "DateTime"
        elif "temperature" in lc and "bme280" in lc:
            rename[c] = "Temperature"
        elif "humidity" in lc:
            rename[c] = "Humidity"
        elif "carbon dioxide" in lc:
            rename[c] = "Carbon Dioxide"
        elif "electrical conductivity" in lc:
            rename[c] = "EC_uS"
        elif "vapor pressure" in lc:
            rename[c] = "VPD_raw"
        elif "ion concent" in lc or lc.startswith("atlas ph"):
            rename[c] = "pH"
    raw = raw.rename(columns=rename)

    cols = [c for c in ["Temperature", "Humidity", "Carbon Dioxide", "EC_uS", "pH"]
            if c in raw.columns]
    daily = _daily_from_timeseries(raw, "DateTime", cols,
                                   cfg["transplant"], cfg["harvest"])

    out = pd.DataFrame(index=daily.index)
    out["T_air_C"] = daily["Temperature"]
    out["T_air_light_C"] = daily["Temperature__light"]
    out["RH_pct"] = daily["Humidity"]
    out["RH_light_pct"] = daily["Humidity__light"]
    out["CO2_ppm"] = daily["Carbon Dioxide"]
    out["P_air_Pa"] = np.nan          # not logged in the Trial 2 export
    out["EC_mS_cm"] = daily["EC_uS"] / 1000.0 if "EC_uS" in daily else np.nan
    out["T_solution_C"] = np.nan      # solution RTD not in this export
    return _finish(out, cfg)


def _finish(out: pd.DataFrame, cfg) -> pd.DataFrame:
    """Add light drivers, fill short gaps, and tidy up."""
    t0 = pd.Timestamp(cfg["transplant"])
    n_days = (pd.Timestamp(cfg["harvest"]) - t0).days + 1
    out = out.reindex(range(n_days))

    out.insert(0, "date", [(t0 + pd.Timedelta(days=int(d))).date()
                           for d in out.index])
    out.index.name = "DAT"

    photoperiod = cfg["photoperiod_h"]
    out["photoperiod_h"] = photoperiod
    out["PPFD_umol_m2_s"] = [ppfd_for_dat(int(d), cfg["ppfd_schedule"])
                             for d in out.index]
    out["DLI_mol_m2_d"] = out["PPFD_umol_m2_s"] * photoperiod * 3600.0 / 1e6

    # Camera/logger outages: interpolate the environment, never the light.
    env_cols = ["T_air_C", "T_air_light_C", "RH_pct", "RH_light_pct",
                "CO2_ppm", "P_air_Pa", "EC_mS_cm", "T_solution_C"]
    n_missing = out[env_cols].isna().all(axis=1).sum()
    out[env_cols] = (out[env_cols].interpolate(limit_direction="both")
                     if n_missing < len(out) else out[env_cols])
    out["P_air_Pa"] = out["P_air_Pa"].fillna(101325.0)

    return out.round(4)


def build_observed() -> pd.DataFrame:
    """Median net fresh weight at the 4-day checkpoints, both trials."""
    raw = pd.read_excel(FW_XLSX, header=None)
    rows = []
    for trial, c0 in ((1, 0), (2, 10)):
        block = raw.iloc[2:10, c0:c0 + 5]
        block.columns = ["DAT", "target_low_g", "target_mid_g",
                         "target_high_g", "observed_median_g"]
        block = block.dropna(subset=["DAT"]).astype(float)
        block.insert(0, "trial", trial)
        rows.append(block)
    obs = pd.concat(rows, ignore_index=True)
    obs["DAT"] = obs["DAT"].astype(int)
    return obs


def build_plant_level() -> pd.DataFrame:
    """Trial 1 per-plant fresh weights: 10 plants x 8 checkpoints.

    Trial 1 kept individual records; Trial 2 appears to have only checkpoint
    medians.  That asymmetry matters downstream -- plant-level data supports
    leave-one-plant-out validation and lets an image trait be matched to the
    plant it belongs to, which a median cannot.

    Note the substrate baseline recorded here is 25.22 g, not the flat 25 g
    the thesis reports subtracting.  The difference is negligible at harvest
    and material at DAT 0.
    """
    raw = pd.read_excel(TRACKER_XLSX, sheet_name="Data_Collection")
    raw = raw.dropna(subset=["Day", "Plant-ID"])
    return pd.DataFrame({
        "trial": 1,
        "DAT": raw["Day"].astype(int),
        "plant_id": raw["Plant-ID"].astype(int),
        "gross_weight_g": raw["Total Fresh Weight (g)"].astype(float),
        "substrate_baseline_g": raw["Baseline (g)"].astype(float),
        "net_fresh_weight_g": raw["New Fresh Weight (g)"].astype(float),
    }).sort_values(["DAT", "plant_id"]).reset_index(drop=True)


def build_ec_log() -> pd.DataFrame:
    """The operator's Trial 1 EC decisions, checkpoint by checkpoint.

    This is the cleanest available statement of what the adaptive regime
    actually did: EC went 1.2-1.4 -> 1.4-1.6 (DAT 8) -> 1.6-1.8 (DAT 12) ->
    1.8-2.0 (DAT 24), each step triggered by the median falling below the
    target band.
    """
    ec = pd.read_excel(TRACKER_XLSX, sheet_name="EC_Log")
    ec = ec.dropna(subset=["Target Mid Range "])
    ec = ec[["Target Fresh Weight Range", "Target Mid Range ",
             "Actual Mid Range", "Status", "EC Range. Now", "New EC"]]
    ec.columns = ["target_band_g", "target_mid_g", "observed_median_g",
                  "status_vs_band", "ec_range_before", "ec_range_after"]
    ec.insert(0, "DAT", list(range(0, 4 * len(ec), 4)))
    return ec.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="print summaries without writing files")
    args = ap.parse_args()

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    frames = {1: build_trial1(), 2: build_trial2()}
    obs = build_observed()

    for trial, df in frames.items():
        print(f"\n=== Trial {trial}: {TRIALS[trial]['label']} ===")
        print(f"{len(df)} days, {df['date'].iloc[0]} -> {df['date'].iloc[-1]}")
        print(df[["T_air_C", "RH_pct", "CO2_ppm", "EC_mS_cm",
                  "DLI_mol_m2_d"]].describe().loc[
                      ["mean", "min", "max"]].round(2).to_string())
        if not args.check:
            path = DATA_OUT / f"trial{trial}_daily_drivers.csv"
            df.to_csv(path)
            print(f"wrote {path}")

    print("\n=== Observed fresh weight checkpoints ===")
    print(obs.to_string(index=False))
    if not args.check:
        path = DATA_OUT / "observed_fresh_weight.csv"
        obs.to_csv(path, index=False)
        print(f"wrote {path}")

    plants = build_plant_level()
    print("\n=== Trial 1 per-plant fresh weight (net, g) ===")
    print(plants.groupby("DAT")["net_fresh_weight_g"]
          .agg(["count", "median", "mean", "min", "max"]).round(2).to_string())
    if not args.check:
        path = DATA_OUT / "trial1_plant_level_fresh_weight.csv"
        plants.to_csv(path, index=False)
        print(f"wrote {path}")

    ec_log = build_ec_log()
    print("\n=== Trial 1 EC steering decisions ===")
    print(ec_log.to_string(index=False))
    if not args.check:
        path = DATA_OUT / "trial1_ec_steering_log.csv"
        ec_log.to_csv(path, index=False)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

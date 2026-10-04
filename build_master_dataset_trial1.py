from pathlib import Path

import numpy as np
import pandas as pd


def _find_col(cols, keyword: str, suffix: str = ""):
    matches = [c for c in cols if (keyword in str(c)) and str(c).endswith(suffix)]
    return matches[0] if matches else None


def _daily_sensor_agg(df: pd.DataFrame, datetime_col: str, feature_cols: dict, prefix: str) -> pd.DataFrame:
    keep = [datetime_col] + [c for c in feature_cols.values() if c is not None]
    tmp = df[keep].copy()
    tmp = tmp.rename(columns={v: k for k, v in feature_cols.items() if v is not None})
    tmp[datetime_col] = pd.to_datetime(tmp[datetime_col], errors="coerce")
    tmp = tmp.dropna(subset=[datetime_col])
    tmp["date"] = tmp[datetime_col].dt.date.astype(str)
    tmp = tmp.drop(columns=[datetime_col])

    numeric_cols = [c for c in tmp.columns if c != "date"]
    for c in numeric_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    agg = tmp.groupby("date")[numeric_cols].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{prefix}_{col}_{stat}" for col, stat in agg.columns]
    return agg.reset_index()


def _daily_actuator_agg(df: pd.DataFrame, datetime_col: str, feature_cols: dict, prefix: str) -> pd.DataFrame:
    keep = [datetime_col] + [c for c in feature_cols.values() if c is not None]
    tmp = df[keep].copy()
    tmp = tmp.rename(columns={v: k for k, v in feature_cols.items() if v is not None})
    tmp[datetime_col] = pd.to_datetime(tmp[datetime_col], errors="coerce")
    tmp = tmp.dropna(subset=[datetime_col])
    tmp["date"] = tmp[datetime_col].dt.date.astype(str)
    tmp = tmp.drop(columns=[datetime_col])

    numeric_cols = [c for c in tmp.columns if c != "date"]
    for c in numeric_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    grouped = tmp.groupby("date")
    out = grouped[numeric_cols].sum(min_count=1).add_prefix(f"{prefix}_").add_suffix("_sum")

    # Additional event counts (how many non-zero events/day)
    for c in numeric_cols:
        out[f"{prefix}_{c}_event_count"] = grouped[c].apply(lambda s: int((s.fillna(0) > 0).sum()))

    return out.reset_index()


def build_sensor_daily_features(project_root: Path) -> pd.DataFrame:
    ambient_path = project_root / "Ambient_nutrient_data.xlsx"

    # Environmental sheet has multi-block columns; row 3 is true header.
    env = pd.read_excel(ambient_path, sheet_name="Environmental Data", header=2)
    env_cols = list(env.columns)

    # Environmental sensor blocks (night/day)
    env_sensor_map_night = {
        "co2_ppm": _find_col(env_cols, "Atlas CO2", ""),
        "air_temp_c": _find_col(env_cols, "BME280 (CH0, Temperature", ""),
        "humidity_pct": _find_col(env_cols, "BME280 (CH1, Humidity", ""),
        "pressure_pa": _find_col(env_cols, "BME280 (CH2, Pressure", ""),
        "dewpoint_c": _find_col(env_cols, "BME280 (CH3, Dewpoint", ""),
        "vpd_pa": _find_col(env_cols, "BME280 (CH5, Vapor Pressure Deficit", ""),
    }
    env_sensor_map_day = {
        "co2_ppm": _find_col(env_cols, "Atlas CO2", ".1"),
        "air_temp_c": _find_col(env_cols, "BME280 (CH0, Temperature", ".1"),
        "humidity_pct": _find_col(env_cols, "BME280 (CH1, Humidity", ".1"),
        "pressure_pa": _find_col(env_cols, "BME280 (CH2, Pressure", ".1"),
        "dewpoint_c": _find_col(env_cols, "BME280 (CH3, Dewpoint", ".1"),
        "vpd_pa": _find_col(env_cols, "BME280 (CH5, Vapor Pressure Deficit", ".1"),
    }

    env_sensor_night = _daily_sensor_agg(env, "DateTime", env_sensor_map_night, "env_night")
    env_sensor_day = _daily_sensor_agg(env, "DateTime.1", env_sensor_map_day, "env_day")

    # Environmental actuator blocks (night/day)
    env_act_map_night = {
        "light_duration_s": _find_col(env_cols, "LIGHT-22", ""),
        "ac_duration_s": _find_col(env_cols, "AC-5", ""),
        "exhaust_duration_s": _find_col(env_cols, "EXHAUST-17", ""),
        "humidifier_duration_s": _find_col(env_cols, "HUMIDIFIER-23", ""),
        "heater_duration_s": _find_col(env_cols, "HEATER-6", ""),
    }
    env_act_map_day = {
        "light_duration_s": _find_col(env_cols, "LIGHT-22", ".1"),
        "ac_duration_s": _find_col(env_cols, "AC-5", ".1"),
        "exhaust_duration_s": _find_col(env_cols, "EXHAUST-17", ".1"),
        "humidifier_duration_s": _find_col(env_cols, "HUMIDIFIER-23", ".1"),
        "heater_duration_s": _find_col(env_cols, "HEATER-6", ".1"),
    }

    env_act_night = _daily_actuator_agg(env, "DateTime.2", env_act_map_night, "env_act_night")
    env_act_day = _daily_actuator_agg(env, "DateTime.3", env_act_map_day, "env_act_day")

    # Nutrient sheet: row 2 is true header.
    nut = pd.read_excel(ambient_path, sheet_name="Nutrient Data", header=1)
    nut_cols = list(nut.columns)

    nut_sensor_map = {
        "ph": _find_col(nut_cols, "Atlas pH", ""),
        "ec_us_cm": _find_col(nut_cols, "Electrical Conductivity", ""),
        "tds_ppm": _find_col(nut_cols, "Total Dissolved Solids", ""),
        "salinity_ppt": _find_col(nut_cols, "Salinity", ""),
        "specific_gravity": _find_col(nut_cols, "Specific Gravity", ""),
        "flow_volume_l": _find_col(nut_cols, "Volume, l", ""),
        "solution_temp_c": _find_col(nut_cols, "PT-1000", ""),
        "flow_rate_l_min": _find_col(nut_cols, "Volume Flow Rate", ""),
    }
    nut_sensor = _daily_sensor_agg(nut, "DateTime", nut_sensor_map, "nutrient")

    nut_act_map = {
        "ph_up_duration_s": _find_col(nut_cols, "pH UP", ""),
        "pump_duration_s": _find_col(nut_cols, "PUMP-27", ""),
        "ph_down_duration_s": _find_col(nut_cols, "pH DOWN", ""),
        "nutrient_ab_duration_s": _find_col(nut_cols, "Nutrient A+B", ""),
        "nutrient_c_duration_s": _find_col(nut_cols, "Nutrient C", ""),
    }
    nut_act = _daily_actuator_agg(nut, "DateTime.1", nut_act_map, "nutrient_act")

    # Merge all daily feature blocks
    dfs = [env_sensor_night, env_sensor_day, env_act_night, env_act_day, nut_sensor, nut_act]
    daily = dfs[0]
    for d in dfs[1:]:
        daily = daily.merge(d, on="date", how="outer")

    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def load_fw_labels_by_day(project_root: Path) -> pd.DataFrame:
    """Median per-plant New Fresh Weight (g) aggregated by experimental Day."""
    tracker_path = project_root / "Lettuce_FW_EC_Tracker_v3_2_.xlsx"
    dc = pd.read_excel(tracker_path, sheet_name="Data_Collection")
    dc["Day"] = pd.to_numeric(dc["Day"], errors="coerce")
    dc["New Fresh Weight (g)"] = pd.to_numeric(dc["New Fresh Weight (g)"], errors="coerce")
    label_by_day = (
        dc.dropna(subset=["Day", "New Fresh Weight (g)"])
        .groupby("Day", as_index=False)["New Fresh Weight (g)"]
        .median()
        .rename(columns={"New Fresh Weight (g)": "target_median_fw_g"})
        .sort_values("Day")
        .reset_index(drop=True)
    )
    label_by_day["Day"] = label_by_day["Day"].astype(int)
    return label_by_day


def build_checkpoint_dataset(
    project_root: Path,
    merged_daily: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    window_days: int = 4,
) -> pd.DataFrame:
    label_by_day = load_fw_labels_by_day(project_root)

    if start_date is None:
        img = pd.read_csv(project_root / "image_features_trial1_singleplant.csv")
        img["date"] = pd.to_datetime(img["date"], errors="coerce")
        start = img["date"].min()
    else:
        start = pd.to_datetime(start_date)

    label_by_day["date"] = (start + pd.to_timedelta(label_by_day["Day"], unit="D")).dt.strftime("%Y-%m-%d")

    # Window aggregate merged_daily features over last 4 days up to checkpoint day
    md = merged_daily.copy()
    md["date"] = pd.to_datetime(md["date"], errors="coerce")
    numeric_cols = [c for c in md.columns if c != "date"]
    for c in numeric_cols:
        md[c] = pd.to_numeric(md[c], errors="coerce")

    rows = []
    for _, r in label_by_day.iterrows():
        day = int(r["Day"])
        checkpoint_date = pd.to_datetime(r["date"])
        window_start = checkpoint_date - pd.Timedelta(days=window_days - 1)
        window = md[(md["date"] >= window_start) & (md["date"] <= checkpoint_date)]

        row = {
            "day": day,
            "date": checkpoint_date.strftime("%Y-%m-%d"),
            "target_median_fw_g": float(r["target_median_fw_g"]),
            "window_start": window_start.strftime("%Y-%m-%d"),
            "window_end": checkpoint_date.strftime("%Y-%m-%d"),
            "window_days_count": int(window["date"].nunique()),
        }
        for c in numeric_cols:
            s = window[c].dropna()
            row[f"{c}_wmean"] = float(s.mean()) if len(s) else np.nan
            row[f"{c}_wstd"] = float(s.std()) if len(s) > 1 else np.nan
            row[f"{c}_wmin"] = float(s.min()) if len(s) else np.nan
            row[f"{c}_wmax"] = float(s.max()) if len(s) else np.nan

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    return out


def main() -> None:
    project_root = Path(__file__).resolve().parent

    # Step 3A: daily sensor features
    sensor_daily = build_sensor_daily_features(project_root)
    sensor_daily.to_csv(project_root / "sensor_features_daily_trial1.csv", index=False)

    # Step 3B: merge with image features
    image_features = pd.read_csv(project_root / "image_features_trial1_singleplant.csv")
    merged_daily = image_features.merge(sensor_daily, on="date", how="left")
    merged_daily.to_csv(project_root / "merged_daily_features_trial1.csv", index=False)

    # Step 3C: checkpoint modeling dataset
    checkpoint = build_checkpoint_dataset(project_root, merged_daily)
    checkpoint.to_csv(project_root / "master_checkpoint_dataset_trial1.csv", index=False)

    print(f"Saved: {project_root / 'sensor_features_daily_trial1.csv'} ({len(sensor_daily)} rows)")
    print(f"Saved: {project_root / 'merged_daily_features_trial1.csv'} ({len(merged_daily)} rows)")
    print(f"Saved: {project_root / 'master_checkpoint_dataset_trial1.csv'} ({len(checkpoint)} rows)")
    print("\nCheckpoint sample:")
    print(checkpoint[["day", "date", "target_median_fw_g", "window_days_count"]].to_string(index=False))


if __name__ == "__main__":
    main()


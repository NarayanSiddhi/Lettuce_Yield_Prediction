#!/usr/bin/env python3
"""
Build dense full-frame (RAW, not segmented) image pairs for Aim 2 generation.

- Source: Timelapse_growtent full JPG frames (timelapse_index.csv)
- Horizon: default +5 calendar days, same clock hour (maximizes pairs; ~16 imgs/day)
- Conditioning: nearest sensors + actuator durations from Mycodo / env+nutrient logs
- Preprocess: center-square crop → resize (default 256), save PNG pairs

Usage:
  python3 ml/forecast_images/build_fullframe_pairs.py --horizon 5 --size 256 --export-images
  python3 ml/forecast_images/build_fullframe_pairs.py --trial trial1 --horizon 5 --size 256 --export-images
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = PROJECT / "Data collection"
INDEX = PROJECT / "segmentation" / "timelapse_index.csv"

SENSOR_COLS_T1 = [
    "Carbon Dioxide",
    "Temperature",
    "Humidity",
    "Pressure",
]
ACT_ENV_COLS = [
    "LIGHT Duration",
    "AC Duration",
    "EXHAUST Duration",
    "HUMIDIFIER Duration",
    "HEATER Duration",
]
ACT_NUT_COLS = [
    "pH Up Duration",
    "PUMP Duration",
    "pH Down Duration",
    "Nutrient AB Duration",
    "Nutrient C Duration",
]
NUT_SENSOR_COLS = [
    "Electrical Conductivity",
    "Volume",
    "Temperature",
    "Volume Flow Rate",
]


def load_index(trial: str | None = None) -> pd.DataFrame:
    idx = pd.read_csv(INDEX)
    idx = idx[idx["usable"] == True].copy()  # noqa: E712
    if trial:
        idx = idx[idx["trial"] == trial]
    idx["date"] = pd.to_datetime(idx["capture_date"]).dt.normalize()
    idx["ts"] = pd.to_datetime(
        idx["capture_date"].astype(str)
        + " "
        + idx["hour"].astype(str).str.zfill(2)
        + ":"
        + idx["minute"].astype(str).str.zfill(2)
        + ":"
        + idx["second"].astype(str).str.zfill(2)
    )
    idx["hour_i"] = idx["hour"].astype(int)
    return idx.sort_values(["trial", "ts"]).reset_index(drop=True)


def load_t1_logs() -> tuple[pd.DataFrame, pd.DataFrame]:
    env = pd.read_csv(DATA / "Trial1_environment_log.csv", parse_dates=["DateTime"])
    nut = pd.read_csv(DATA / "Trial1_nutrient_log.csv", parse_dates=["DateTime"])
    env = env.sort_values("DateTime")
    nut = nut.sort_values("DateTime")
    return env, nut


def load_t2_mycodo() -> pd.DataFrame:
    p = DATA / "Trial2_Mycodo_raw_export.xlsx"
    df = pd.read_excel(p, sheet_name=0)
    df = df.rename(
        columns={
            "DateTime": "DateTime",
            "Atlas CO2 (Carbon Dioxide Gas) (CH0, Carbon Dioxide, ppm)": "CO2_ppm",
            "Atlas EC (CH0, Electrical Conductivity, μS/cm)": "EC_uScm",
            "BME280 (CH0, Temperature, °C)": "Temperature",
            "BME280 (CH1, Humidity, %)": "Humidity",
            "BME280 (CH5, Vapor Pressure Deficit, Pa)": "VPD_Pa",
            "Atlas pH (CH0, Ion Concentration, pH)": "pH",
        }
    )
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    return df.sort_values("DateTime").reset_index(drop=True)


def nearest_row(df: pd.DataFrame, ts: pd.Timestamp, max_min: float = 45.0) -> pd.Series | None:
    if df.empty:
        return None
    # asof merge style
    pos = df["DateTime"].searchsorted(ts)
    cands = []
    for i in (pos - 1, pos):
        if 0 <= i < len(df):
            cands.append(df.iloc[i])
    if not cands:
        return None
    best = min(cands, key=lambda r: abs((r["DateTime"] - ts).total_seconds()))
    if abs((best["DateTime"] - ts).total_seconds()) > max_min * 60:
        return None
    return best


def window_means(df: pd.DataFrame, ts: pd.Timestamp, hours: float, cols: list[str]) -> dict:
    lo = ts - timedelta(hours=hours)
    sub = df[(df["DateTime"] >= lo) & (df["DateTime"] <= ts)]
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(sub[c], errors="coerce")
        out[f"{c}_mean{int(hours)}h"] = float(s.mean()) if s.notna().any() else float("nan")
        out[f"{c}_sum{int(hours)}h"] = float(s.sum()) if s.notna().any() else float("nan")
    return out


def features_at_time(trial: str, ts: pd.Timestamp, caches: dict) -> dict:
    """Sensors + actuators around image timestamp."""
    feat: dict = {"trial": trial, "ts": str(ts)}
    if trial == "trial1":
        env, nut = caches["t1_env"], caches["t1_nut"]
        er = nearest_row(env, ts)
        nr = nearest_row(nut, ts)
        if er is not None:
            for c in SENSOR_COLS_T1 + ACT_ENV_COLS:
                if c in er.index:
                    feat[f"env_{c.replace(' ', '_')}"] = float(pd.to_numeric(er[c], errors="coerce"))
            feat.update(
                {
                    f"env_{k}": v
                    for k, v in window_means(env, ts, 24, ACT_ENV_COLS + SENSOR_COLS_T1).items()
                }
            )
        if nr is not None:
            for c in ACT_NUT_COLS + NUT_SENSOR_COLS:
                if c in nr.index:
                    safe = c.replace(" ", "_")
                    feat[f"nut_{safe}"] = float(pd.to_numeric(nr[c], errors="coerce"))
            feat.update(
                {
                    f"nut_{k}": v
                    for k, v in window_means(nut, ts, 24, ACT_NUT_COLS + NUT_SENSOR_COLS).items()
                }
            )
    else:
        my = caches["t2_my"]
        r = nearest_row(my, ts)
        if r is not None:
            for c in ["CO2_ppm", "EC_uScm", "Temperature", "Humidity", "VPD_Pa", "pH"]:
                if c in r.index:
                    feat[f"my_{c}"] = float(pd.to_numeric(r[c], errors="coerce"))
            feat.update(
                {
                    f"my_{k}": v
                    for k, v in window_means(
                        my, ts, 24, ["CO2_ppm", "EC_uScm", "Temperature", "Humidity", "VPD_Pa", "pH"]
                    ).items()
                }
            )
        # daily drivers fallback extras
        drivers = caches["t2_drivers"]
        day = pd.Timestamp(ts.normalize())
        hit = drivers[drivers["date"] == day]
        if len(hit):
            for c in ["PPFD_umol_m2_s", "DLI_mol_m2_d", "photoperiod_h", "EC_mS_cm", "T_air_C"]:
                if c in hit.columns:
                    feat[f"day_{c}"] = float(hit.iloc[0][c])
    return feat


def maximize_same_hour_pairs(g: pd.DataFrame, horizon_days: int) -> list[tuple[pd.Series, pd.Series]]:
    """Greedy 1:1 pairs: same hour, date_f = date_t + horizon."""
    pairs = []
    # group frames by (date, hour)
    buckets: dict[tuple, list[pd.Series]] = {}
    for _, r in g.iterrows():
        key = (pd.Timestamp(r["date"]), int(r["hour_i"]))
        buckets.setdefault(key, []).append(r)
    for key, rows in buckets.items():
        rows = sorted(rows, key=lambda r: r["ts"])
        buckets[key] = rows

    used_f = set()
    for (date_t, hour), rows_t in sorted(buckets.items()):
        date_f = date_t + timedelta(days=horizon_days)
        rows_f = buckets.get((date_f, hour), [])
        if not rows_f:
            continue
        # pair in time order
        jf = 0
        for rt in rows_t:
            while jf < len(rows_f):
                rf = rows_f[jf]
                jf += 1
                uid = (rf["source_path"], str(rf["ts"]))
                if uid in used_f:
                    continue
                used_f.add(uid)
                pairs.append((rt, rf))
                break
    return pairs


def center_square_resize(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.BICUBIC)


def build_trial(trial: str, horizon: int, size: int, export: bool, caches: dict) -> pd.DataFrame:
    idx = load_index(trial)
    print(f"[{trial}] usable frames={len(idx)} days={idx['date'].nunique()}")
    raw_pairs = maximize_same_hour_pairs(idx, horizon)
    print(f"[{trial}] same-hour +{horizon}d pairs={len(raw_pairs)}")

    rows = []
    ds_root = OUT / "dataset" / f"fullframe_{trial}_h{horizon}_{size}"
    if export:
        (ds_root / "train" / "input").mkdir(parents=True, exist_ok=True)
        (ds_root / "train" / "target").mkdir(parents=True, exist_ok=True)
        (ds_root / "val" / "input").mkdir(parents=True, exist_ok=True)
        (ds_root / "val" / "target").mkdir(parents=True, exist_ok=True)
        (ds_root / "test" / "input").mkdir(parents=True, exist_ok=True)
        (ds_root / "test" / "target").mkdir(parents=True, exist_ok=True)

    # cup-free split by calendar day of target (leave late days for test)
    dates_f = sorted({pd.Timestamp(rf["date"]) for _, rf in raw_pairs})
    if len(dates_f) >= 6:
        test_dates = set(dates_f[-2:])
        val_dates = set(dates_f[-4:-2])
    else:
        test_dates = set(dates_f[-1:]) if dates_f else set()
        val_dates = set(dates_f[-2:-1]) if len(dates_f) > 1 else set()

    for i, (rt, rf) in enumerate(raw_pairs):
        feat = features_at_time(trial, pd.Timestamp(rt["ts"]), caches)
        feat.update(
            {
                "pair_id": i,
                "trial": trial,
                "date_t": str(pd.Timestamp(rt["date"]).date()),
                "date_f": str(pd.Timestamp(rf["date"]).date()),
                "hour": int(rt["hour_i"]),
                "ts_t": str(rt["ts"]),
                "ts_f": str(rf["ts"]),
                "source_t": rt["source_path"],
                "source_f": rf["source_path"],
                "horizon_days": horizon,
            }
        )
        d_f = pd.Timestamp(rf["date"])
        if d_f in test_dates:
            split = "test"
        elif d_f in val_dates:
            split = "val"
        else:
            split = "train"
        feat["split"] = split

        stem = f"{pd.Timestamp(rt['date']).strftime('%Y%m%d')}_h{int(rt['hour_i']):02d}_to_{pd.Timestamp(rf['date']).strftime('%Y%m%d')}"
        # disambiguate multiples
        stem = f"{stem}_{i:04d}"
        if export:
            src_t = PROJECT / rt["source_path"]
            src_f = PROJECT / rf["source_path"]
            if not src_t.exists() or not src_f.exists():
                print(f"  missing file skip {stem}")
                continue
            try:
                im_t = center_square_resize(Image.open(src_t).convert("RGB"), size)
                im_f = center_square_resize(Image.open(src_f).convert("RGB"), size)
            except Exception as e:
                print(f"  read fail {stem}: {e}")
                continue
            rel_in = f"ml/forecast_images/dataset/fullframe_{trial}_h{horizon}_{size}/{split}/input/{stem}.png"
            rel_tg = f"ml/forecast_images/dataset/fullframe_{trial}_h{horizon}_{size}/{split}/target/{stem}.png"
            im_t.save(PROJECT / rel_in)
            im_f.save(PROJECT / rel_tg)
            feat["input_png"] = rel_in
            feat["target_png"] = rel_tg
        rows.append(feat)
        if (i + 1) % 50 == 0:
            print(f"  [{trial}] processed {i+1}/{len(raw_pairs)}", flush=True)

    df = pd.DataFrame(rows)
    out_csv = OUT / f"pairs_fullframe_{trial}_h{horizon}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[{trial}] wrote {out_csv} n={len(df)} split={df['split'].value_counts().to_dict()}")

    if export and len(df):
        # climate norm from train numeric cols
        num_cols = [
            c
            for c in df.columns
            if c.startswith(("env_", "nut_", "my_", "day_")) and pd.api.types.is_numeric_dtype(df[c])
        ]
        train = df[df["split"] == "train"]
        norm = {}
        for c in num_cols:
            s = pd.to_numeric(train[c], errors="coerce")
            norm[c] = {"mean": float(s.mean()) if s.notna().any() else 0.0, "std": float(s.std()) if s.notna().sum() > 1 else 1.0}
            if norm[c]["std"] == 0 or not np.isfinite(norm[c]["std"]):
                norm[c]["std"] = 1.0
        meta = {
            "trial": trial,
            "horizon_days": horizon,
            "size": size,
            "n_pairs": len(df),
            "splits": df["split"].value_counts().to_dict(),
            "climate": norm,
            "feature_cols": num_cols,
        }
        ds_root = OUT / "dataset" / f"fullframe_{trial}_h{horizon}_{size}"
        (ds_root / "climate_norm.json").write_text(json.dumps(meta, indent=2))
        df.to_csv(ds_root / "manifest.csv", index=False)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trial", choices=["trial1", "trial2", "both"], default="both")
    p.add_argument("--horizon", type=int, default=5, help="Days between paired frames (same hour)")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--export-images", action="store_true")
    args = p.parse_args()

    caches = {}
    print("Loading sensor/actuator logs…")
    caches["t1_env"], caches["t1_nut"] = load_t1_logs()
    caches["t2_my"] = load_t2_mycodo()
    d2 = pd.read_csv(PROJECT / "Lettuce_model" / "data" / "trial2_daily_drivers.csv")
    d2["date"] = pd.to_datetime(d2["date"]).dt.normalize()
    caches["t2_drivers"] = d2

    trials = ["trial1", "trial2"] if args.trial == "both" else [args.trial]
    all_dfs = []
    for t in trials:
        all_dfs.append(build_trial(t, args.horizon, args.size, args.export_images, caches))
    if len(all_dfs) > 1:
        both = pd.concat(all_dfs, ignore_index=True)
        both.to_csv(OUT / f"pairs_fullframe_both_h{args.horizon}.csv", index=False)
        print(f"Combined pairs={len(both)}")


if __name__ == "__main__":
    main()

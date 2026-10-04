#!/usr/bin/env python3
"""
Build dense per-cup multimodal pairs for Aim 2 generation.

For each keep+cup_id crop at time t, pair with the same cup_id at t+horizon
(same clock hour). Always attach sensors + actuators at t.

Two image modes (exported as separate datasets, both trained later):
  rgb  — bbox crop from the ORIGINAL raw tray JPG
  seg  — FastSAM RGBA crop composited plant-on-black (alpha = mask)

Take: dense same-hour pairing + climate conditioning from full-frame pipeline;
      cup_id tracking from FastSAM / assign_cup_ids.
Skip: whole-tray generation as the primary target.
Improve: learn single-plant growth with more samples and plant-focused pixels.

Usage:
  python3 ml/forecast_images/build_cup_multimodal_pairs.py --trial both --horizon 5 --size 128 --export-images
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CROPS_CSV = PROJECT / "segmentation" / "crops_with_cups.csv"

# Reuse sensor/actuator feature builders from full-frame pipeline
sys.path.insert(0, str(OUT))
from build_fullframe_pairs import (  # noqa: E402
    features_at_time,
    load_t1_logs,
    load_t2_mycodo,
)

TS_RE = re.compile(r"plant-(\d{8})-(\d{6})")


def parse_ts(source_path: str) -> pd.Timestamp:
    m = TS_RE.search(str(source_path))
    if not m:
        return pd.NaT
    return pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M%S")


def load_cup_table(
    trial: str | None,
    max_bbox_side: int,
    min_plant_frac: float,
) -> pd.DataFrame:
    df = pd.read_csv(CROPS_CSV)
    df = df[(df["keep"] == True) & df["cup_id"].notna()].copy()  # noqa: E712
    if trial:
        df = df[df["trial"] == trial]
    df["cup_id"] = df["cup_id"].astype(int)
    df["bw"] = df["bbox_x2"] - df["bbox_x1"]
    df["bh"] = df["bbox_y2"] - df["bbox_y1"]
    df = df[(df[["bw", "bh"]].max(axis=1) <= max_bbox_side) & (df["plant_color_frac"] >= min_plant_frac)]
    df["ts"] = df["source_path"].map(parse_ts)
    df = df[df["ts"].notna()].copy()
    df["date"] = df["ts"].dt.normalize()
    df["hour_i"] = df["ts"].dt.hour
    # one observation per cup × date × hour (prefer greener / larger plant signal)
    df = (
        df.sort_values(["plant_color_frac", "mask_area_px"], ascending=[False, False])
        .groupby(["trial", "cup_id", "date", "hour_i"], as_index=False)
        .first()
    )
    return df.sort_values(["trial", "cup_id", "ts"]).reset_index(drop=True)


def assign_cup_splits(cups: list[int], pair_counts: dict[int, int] | None = None) -> dict[str, list[int]]:
    """
    Prefer putting high-pair-count cups in train so the generator sees enough growth
    examples; hold out 2 cups for val and 2 for test when possible.
    """
    cups = sorted(cups)
    if pair_counts is None:
        pair_counts = {c: 1 for c in cups}
    # most pairs first → train
    ordered = sorted(cups, key=lambda c: (-pair_counts.get(c, 0), c))
    if len(ordered) < 3:
        return {"train": ordered[:1], "val": ordered[1:2], "test": ordered[2:3] or ordered[-1:]}
    if len(ordered) == 3:
        return {"train": ordered[:1], "val": ordered[1:2], "test": ordered[2:3]}
    if len(ordered) == 4:
        return {"train": ordered[:2], "val": ordered[2:3], "test": ordered[3:4]}
    # >=5 cups: 2 test, 2 val, rest train (train = highest counts)
    test_cups = ordered[-2:]
    val_cups = ordered[-4:-2]
    train_cups = ordered[:-4]
    return {"train": sorted(train_cups), "val": sorted(val_cups), "test": sorted(test_cups)}


def maximize_cup_pairs(g: pd.DataFrame, horizon_days: int) -> list[tuple[pd.Series, pd.Series]]:
    """Same cup, same hour, date_f = date_t + horizon."""
    buckets: dict[tuple, list[pd.Series]] = {}
    for _, r in g.iterrows():
        key = (pd.Timestamp(r["date"]), int(r["hour_i"]))
        buckets.setdefault(key, []).append(r)
    for key, rows in buckets.items():
        buckets[key] = sorted(rows, key=lambda r: r["ts"])

    pairs = []
    used_f: set[tuple] = set()
    for (date_t, hour), rows_t in sorted(buckets.items()):
        date_f = date_t + timedelta(days=horizon_days)
        rows_f = buckets.get((date_f, hour), [])
        if not rows_f:
            continue
        jf = 0
        for rt in rows_t:
            while jf < len(rows_f):
                rf = rows_f[jf]
                jf += 1
                uid = (int(rf["cup_id"]), str(rf["ts"]), str(rf["crop_path"]))
                if uid in used_f:
                    continue
                used_f.add(uid)
                pairs.append((rt, rf))
                break
    return pairs


def resize_rgb(img: Image.Image, size: int) -> Image.Image:
    return img.convert("RGB").resize((size, size), Image.BICUBIC)


def crop_original_rgb(row: pd.Series, size: int) -> Image.Image | None:
    src = PROJECT / str(row["source_path"])
    if not src.exists():
        return None
    try:
        im = Image.open(src).convert("RGB")
    except Exception:
        return None
    x1, y1, x2, y2 = int(row["bbox_x1"]), int(row["bbox_y1"]), int(row["bbox_x2"]), int(row["bbox_y2"])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(im.width, x2), min(im.height, y2)
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return None
    return resize_rgb(im.crop((x1, y1, x2, y2)), size)


def seg_plant_on_black(row: pd.Series, size: int) -> Image.Image | None:
    cp = PROJECT / str(row["crop_path"])
    if not cp.exists():
        return None
    try:
        rgba = np.asarray(Image.open(cp).convert("RGBA"))
    except Exception:
        return None
    rgb = rgba[:, :, :3].astype(np.float32)
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    comp = (rgb * a).clip(0, 255).astype(np.uint8)
    return resize_rgb(Image.fromarray(comp, mode="RGB"), size)


def export_mode_images(
    rows: list[dict],
    mode: str,
    trial: str,
    horizon: int,
    size: int,
) -> list[dict]:
    ds = OUT / "dataset" / f"cup_{mode}_{trial}_h{horizon}_{size}"
    for split in ("train", "val", "test"):
        (ds / split / "input").mkdir(parents=True, exist_ok=True)
        (ds / split / "target").mkdir(parents=True, exist_ok=True)

    out_rows = []
    for i, feat in enumerate(rows):
        rt = feat["_rt"]
        rf = feat["_rf"]
        split = feat["split"]
        stem = (
            f"cup{int(feat['cup_id']):02d}_"
            f"{pd.Timestamp(feat['date_t']).strftime('%Y%m%d')}_h{int(feat['hour']):02d}_"
            f"to_{pd.Timestamp(feat['date_f']).strftime('%Y%m%d')}_{i:05d}"
        )
        if mode == "rgb":
            im_t = crop_original_rgb(rt, size)
            im_f = crop_original_rgb(rf, size)
        else:
            im_t = seg_plant_on_black(rt, size)
            im_f = seg_plant_on_black(rf, size)
        if im_t is None or im_f is None:
            continue
        rel_in = f"ml/forecast_images/dataset/cup_{mode}_{trial}_h{horizon}_{size}/{split}/input/{stem}.png"
        rel_tg = f"ml/forecast_images/dataset/cup_{mode}_{trial}_h{horizon}_{size}/{split}/target/{stem}.png"
        im_t.save(PROJECT / rel_in)
        im_f.save(PROJECT / rel_tg)
        row = {k: v for k, v in feat.items() if not k.startswith("_")}
        row["mode"] = mode
        row["input_png"] = rel_in
        row["target_png"] = rel_tg
        out_rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  [{trial}/{mode}] exported {i+1}/{len(rows)}", flush=True)
    return out_rows


def write_climate_norm(df: pd.DataFrame, ds_root: Path, trial: str, horizon: int, size: int, mode: str, split_map: dict):
    num_cols = [
        c
        for c in df.columns
        if c.startswith(("env_", "nut_", "my_", "day_")) and pd.api.types.is_numeric_dtype(df[c])
    ]
    train = df[df["split"] == "train"]
    norm = {}
    for c in num_cols:
        s = pd.to_numeric(train[c], errors="coerce")
        std = float(s.std()) if s.notna().sum() > 1 else 1.0
        if not np.isfinite(std) or std == 0:
            std = 1.0
        norm[c] = {
            "mean": float(s.mean()) if s.notna().any() else 0.0,
            "std": std,
        }
    meta = {
        "trial": trial,
        "mode": mode,
        "horizon_days": horizon,
        "size": size,
        "n_pairs": len(df),
        "splits": df["split"].value_counts().to_dict(),
        "train_cups": split_map["train"],
        "val_cups": split_map["val"],
        "test_cups": split_map["test"],
        "climate": norm,
        "feature_cols": num_cols,
        "conditioning": "sensors_and_actuators_required",
    }
    (ds_root / "climate_norm.json").write_text(json.dumps(meta, indent=2))
    df.to_csv(ds_root / "manifest.csv", index=False)


def build_trial(
    trial: str,
    horizon: int,
    size: int,
    export: bool,
    modes: list[str],
    caches: dict,
    max_bbox_side: int,
    min_plant_frac: float,
) -> pd.DataFrame:
    cups = load_cup_table(trial, max_bbox_side, min_plant_frac)
    print(
        f"[{trial}] cup-date-hour rows={len(cups)} cups={cups.cup_id.nunique()} days={cups.date.nunique()}",
        flush=True,
    )

    raw_pairs: list[tuple[pd.Series, pd.Series]] = []
    for cup_id, g in cups.groupby("cup_id"):
        raw_pairs.extend(maximize_cup_pairs(g, horizon))
    print(f"[{trial}] same-cup same-hour +{horizon}d pairs={len(raw_pairs)}", flush=True)

    paired_cups = sorted({int(rt["cup_id"]) for rt, _ in raw_pairs})
    pair_counts: dict[int, int] = {}
    for rt, _ in raw_pairs:
        cid = int(rt["cup_id"])
        pair_counts[cid] = pair_counts.get(cid, 0) + 1
    split_map = assign_cup_splits(paired_cups, pair_counts)
    cup_to_split = {c: s for s, ids in split_map.items() for c in ids}
    print(f"[{trial}] cup splits (by pair-count) {split_map}", flush=True)
    print(f"[{trial}] pairs/cup {dict(sorted(pair_counts.items()))}", flush=True)

    feat_rows = []
    for i, (rt, rf) in enumerate(raw_pairs):
        feat = features_at_time(trial, pd.Timestamp(rt["ts"]), caches)
        feat.update(
            {
                "pair_id": i,
                "trial": trial,
                "cup_id": int(rt["cup_id"]),
                "date_t": str(pd.Timestamp(rt["date"]).date()),
                "date_f": str(pd.Timestamp(rf["date"]).date()),
                "hour": int(rt["hour_i"]),
                "ts_t": str(rt["ts"]),
                "ts_f": str(rf["ts"]),
                "source_t": rt["source_path"],
                "source_f": rf["source_path"],
                "crop_t": rt["crop_path"],
                "crop_f": rf["crop_path"],
                "bbox_t": f"{int(rt['bbox_x1'])},{int(rt['bbox_y1'])},{int(rt['bbox_x2'])},{int(rt['bbox_y2'])}",
                "bbox_f": f"{int(rf['bbox_x1'])},{int(rf['bbox_y1'])},{int(rf['bbox_x2'])},{int(rf['bbox_y2'])}",
                "mask_area_t": float(rt["mask_area_px"]),
                "mask_area_f": float(rf["mask_area_px"]),
                "horizon_days": horizon,
                "split": cup_to_split[int(rt["cup_id"])],
                "_rt": rt,
                "_rf": rf,
            }
        )
        feat_rows.append(feat)

    # CSV without image paths / internal series
    csv_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in feat_rows]
    base_df = pd.DataFrame(csv_rows)
    out_csv = OUT / f"pairs_cup_{trial}_h{horizon}.csv"
    base_df.to_csv(out_csv, index=False)
    print(f"[{trial}] wrote {out_csv} n={len(base_df)}", flush=True)

    if not export:
        return base_df

    all_exported = []
    for mode in modes:
        print(f"[{trial}] exporting mode={mode} …", flush=True)
        exported = export_mode_images(feat_rows, mode, trial, horizon, size)
        edf = pd.DataFrame(exported)
        if edf.empty:
            print(f"[{trial}/{mode}] WARNING: no images exported", flush=True)
            continue
        ds_root = OUT / "dataset" / f"cup_{mode}_{trial}_h{horizon}_{size}"
        write_climate_norm(edf, ds_root, trial, horizon, size, mode, split_map)
        mode_csv = OUT / f"pairs_cup_{mode}_{trial}_h{horizon}.csv"
        edf.to_csv(mode_csv, index=False)
        print(
            f"[{trial}/{mode}] exported n={len(edf)} splits={edf['split'].value_counts().to_dict()} → {ds_root}",
            flush=True,
        )
        all_exported.append(edf)
    if all_exported:
        return pd.concat(all_exported, ignore_index=True)
    return base_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trial", choices=["trial1", "trial2", "both"], default="both")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--export-images", action="store_true")
    p.add_argument(
        "--modes",
        default="rgb,seg",
        help="Comma-separated: rgb (original bbox crop), seg (FastSAM plant-on-black)",
    )
    p.add_argument("--max-bbox-side", type=int, default=900)
    p.add_argument("--min-plant-frac", type=float, default=0.35)
    args = p.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in ("rgb", "seg"):
            raise SystemExit(f"Unknown mode {m}; use rgb and/or seg")

    print("Loading sensor/actuator logs…", flush=True)
    t1_env, t1_nut = load_t1_logs()
    caches = {
        "t1_env": t1_env,
        "t1_nut": t1_nut,
        "t2_my": load_t2_mycodo(),
    }
    d2 = pd.read_csv(PROJECT / "Lettuce_model" / "data" / "trial2_daily_drivers.csv")
    d2["date"] = pd.to_datetime(d2["date"]).dt.normalize()
    caches["t2_drivers"] = d2

    trials = ["trial1", "trial2"] if args.trial == "both" else [args.trial]
    dfs = []
    for t in trials:
        dfs.append(
            build_trial(
                t,
                args.horizon,
                args.size,
                args.export_images,
                modes,
                caches,
                args.max_bbox_side,
                args.min_plant_frac,
            )
        )
    if len(dfs) > 1:
        both = pd.concat(dfs, ignore_index=True)
        both.to_csv(OUT / f"pairs_cup_both_h{args.horizon}.csv", index=False)
        print(f"Combined rows={len(both)}", flush=True)


if __name__ == "__main__":
    main()

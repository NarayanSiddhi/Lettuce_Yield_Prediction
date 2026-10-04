"""
Build cup image pairs for Pix2Pix: crop_t + climate → crop_{t+4}.

Writes:
  ml/forecast_images/pairs_{trial}.csv
  ml/forecast_images/dataset/{trial}_{size}/  (optional resized PNG pairs)

Usage:
  python3 ml/forecast_images/build_pix2pix_pairs.py --trial trial2 --size 128 --export-images
  python3 ml/forecast_images/build_pix2pix_pairs.py --trial trial1 --size 128 --export-images
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
HORIZON = 4
CLIMATE_COLS = [
    "T_air_C",
    "RH_pct",
    "CO2_ppm",
    "EC_mS_cm",
    "PPFD_umol_m2_s",
    "DLI_mol_m2_d",
]


def load_drivers(trial: str = "trial1") -> pd.DataFrame:
    d = pd.read_csv(PROJECT / "Lettuce_model" / "data" / f"{trial}_daily_drivers.csv")
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    keep = ["DAT", "date"] + [c for c in CLIMATE_COLS if c in d.columns]
    return d[keep]


def climate_vec(drivers: pd.DataFrame, dat: int, window: int = 4) -> dict[str, float]:
    sub = drivers[(drivers["DAT"] >= dat - window + 1) & (drivers["DAT"] <= dat)]
    out: dict[str, float] = {}
    for c in CLIMATE_COLS:
        if c not in drivers.columns:
            continue
        now = drivers.loc[drivers["DAT"] == dat, c]
        out[f"{c}_now"] = float(now.iloc[0]) if len(now) else float("nan")
        out[f"{c}_mean{window}d"] = float(sub[c].mean()) if len(sub) else float("nan")
    return out


def resolve_crop(path_str: str) -> Path | None:
    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT / p
    return p if p.exists() else None


def build_pairs(trial: str = "trial1") -> pd.DataFrame:
    cup = pd.read_csv(PROJECT / "segmentation" / "daily_cup_features.csv")
    cup = cup[cup["trial"] == trial].copy()
    cup["date"] = pd.to_datetime(cup["date"]).dt.normalize()
    cup = (
        cup.sort_values(["cup_id", "date", "selection_score"], ascending=[True, True, False])
        .groupby(["cup_id", "date"], as_index=False)
        .first()
    )
    drivers = load_drivers(trial)
    cup = cup.merge(drivers[["date", "DAT"]], on="date", how="inner")

    rows = []
    for cup_id, g in cup.groupby("cup_id"):
        by_dat = {int(r["DAT"]): r for _, r in g.iterrows()}
        for dat_t, row_t in by_dat.items():
            dat_f = dat_t + HORIZON
            if dat_f not in by_dat:
                continue
            row_f = by_dat[dat_f]
            src = resolve_crop(str(row_t["crop_path"]))
            tgt = resolve_crop(str(row_f["crop_path"]))
            if src is None or tgt is None:
                continue
            feat = {
                "trial": trial,
                "cup_id": int(cup_id),
                "day_t": dat_t,
                "day_future": dat_f,
                "date_t": str(pd.Timestamp(row_t["date"]).date()),
                "date_future": str(pd.Timestamp(row_f["date"]).date()),
                "crop_t": str(src.relative_to(PROJECT)),
                "crop_future": str(tgt.relative_to(PROJECT)),
                "mask_area_t": float(row_t["mask_area_px"]),
                "mask_area_future": float(row_f["mask_area_px"]),
            }
            feat.update(climate_vec(drivers, dat_t))
            rows.append(feat)
    return pd.DataFrame(rows)


def assign_cup_splits(cups: list[int]) -> dict[str, list[int]]:
    """
    Split by cup_id so the same plant never appears in two splits.
    Default: test = highest two cup ids present; val = next two; train = rest.
    """
    cups = sorted(cups)
    if len(cups) < 3:
        # tiny fallback
        return {"train": cups[:1], "val": cups[1:2], "test": cups[2:3] or cups[-1:]}
    test_cups = cups[-2:]
    remaining = cups[:-2]
    if len(remaining) == 1:
        val_cups = remaining
        train_cups = []
    elif len(remaining) == 2:
        val_cups = remaining[-1:]
        train_cups = remaining[:-1]
    else:
        val_cups = remaining[-2:]
        train_cups = remaining[:-2]
    return {"train": train_cups, "val": val_cups, "test": test_cups}


def export_images(pairs: pd.DataFrame, size: int, trial: str) -> None:
    ds = OUT_DIR / "dataset" / f"{trial}_{size}"
    for split in ("train", "val", "test"):
        (ds / split / "input").mkdir(parents=True, exist_ok=True)
        (ds / split / "target").mkdir(parents=True, exist_ok=True)

    cups = sorted(int(c) for c in pairs["cup_id"].unique())
    split_map = assign_cup_splits(cups)
    cup_to_split = {}
    for split, ids in split_map.items():
        for c in ids:
            cup_to_split[c] = split

    meta_rows = []
    for _, r in pairs.iterrows():
        split = cup_to_split[int(r["cup_id"])]
        stem = f"cup{int(r['cup_id']):02d}_d{int(r['day_t']):02d}_to_d{int(r['day_future']):02d}"
        in_path = ds / split / "input" / f"{stem}.png"
        tg_path = ds / split / "target" / f"{stem}.png"
        Image.open(PROJECT / r["crop_t"]).convert("RGB").resize((size, size), Image.BICUBIC).save(in_path)
        Image.open(PROJECT / r["crop_future"]).convert("RGB").resize((size, size), Image.BICUBIC).save(tg_path)
        meta = dict(r)
        meta["split"] = split
        meta["input_png"] = str(in_path.relative_to(PROJECT))
        meta["target_png"] = str(tg_path.relative_to(PROJECT))
        meta_rows.append(meta)

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(ds / "manifest.csv", index=False)

    # Normalize climate using TRAIN only (no leakage into val/test)
    clim_cols = [c for c in meta_df.columns if c.endswith("_now") or "mean4d" in c]
    train_df = meta_df[meta_df["split"] == "train"]
    stats = {}
    for c in clim_cols:
        stats[c] = {
            "mean": float(train_df[c].mean()),
            "std": float(train_df[c].std() or 1.0),
        }
    (ds / "climate_norm.json").write_text(
        json.dumps(
            {
                "trial": trial,
                "train_cups": split_map["train"],
                "val_cups": split_map["val"],
                "test_cups": split_map["test"],
                "climate": stats,
            },
            indent=2,
        )
    )
    print(f"Exported images to {ds}")
    for split in ("train", "val", "test"):
        n = int((meta_df["split"] == split).sum())
        print(f"  {split}: {n} pairs  cups={split_map[split]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", choices=["trial1", "trial2"], default="trial1")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--export-images", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(args.trial)
    out_csv = OUT_DIR / f"pairs_{args.trial}.csv"
    pairs.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(pairs)} pairs, {pairs.cup_id.nunique()} cups)")
    if args.export_images:
        export_images(pairs, args.size, args.trial)


if __name__ == "__main__":
    main()

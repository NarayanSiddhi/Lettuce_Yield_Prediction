"""
Preprocess cup crops for future-image training and export ControlNet-ready pairs.

Steps:
  - RGBA → RGB on white
  - vegetation bbox crop (ExGR) with padding
  - drop near-empty crops
  - resize to square
  - keep same cup-level train/val/test split as Pix2Pix

Usage:
  python3 ml/forecast_images/preprocess_and_export.py --trial trial1 --size 256
  python3 ml/forecast_images/preprocess_and_export.py --trial trial2 --size 256
  python3 ml/forecast_images/preprocess_and_export.py --trial both --size 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(OUT_DIR))
from build_pix2pix_pairs import assign_cup_splits, build_pairs  # noqa: E402


def rgba_to_rgb_white(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def vegetation_mask_bgr(bgr: np.ndarray) -> np.ndarray:
    """ExGR-ish vegetation mask; robust enough for tent crops."""
    img = bgr.astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    exgr = 2.0 * g - r - b
    # also keep green-ish HSV as backup
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hsv_m = ((h >= 25) & (h <= 95) & (s >= 25) & (v >= 30)).astype(np.uint8)
    ex_m = (exgr > 15).astype(np.uint8)
    m = np.clip(ex_m + hsv_m, 0, 1).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return m


def tight_square_crop(rgb: Image.Image, pad_frac: float = 0.12) -> tuple[Image.Image, dict]:
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    m = vegetation_mask_bgr(bgr)
    ys, xs = np.where(m > 0)
    h, w = m.shape
    veg_frac = float(m.mean())
    info = {"veg_frac": veg_frac, "used_full_frame": False}
    if len(xs) < 30 or veg_frac < 0.01:
        info["used_full_frame"] = True
        return rgb, info

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    side = int(max(bw, bh) * (1.0 + 2 * pad_frac))
    side = max(side, 32)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0n = max(0, cx - side // 2)
    y0n = max(0, cy - side // 2)
    x1n = min(w, x0n + side)
    y1n = min(h, y0n + side)
    x0n = max(0, x1n - side)
    y0n = max(0, y1n - side)
    crop = rgb.crop((x0n, y0n, x1n, y1n))
    info.update({"bbox": [x0n, y0n, x1n, y1n], "side": side})
    return crop, info


def climate_prompt(row: pd.Series) -> str:
    def g(key: str, nd: int = 1) -> str:
        try:
            return f"{float(row[key]):.{nd}f}"
        except Exception:
            return "?"

    return (
        "top-down photo of a lettuce plant in a hydroponic grow tent cup, "
        f"DAT {int(row['day_t'])} to DAT {int(row['day_future'])}, "
        f"air {g('T_air_C_mean4d')}C, RH {g('RH_pct_mean4d')}%, "
        f"CO2 {g('CO2_ppm_mean4d', 0)} ppm, EC {g('EC_mS_cm_mean4d', 2)} mS/cm, "
        f"PPFD {g('PPFD_umol_m2_s_mean4d', 0)}, DLI {g('DLI_mol_m2_d_mean4d', 1)}"
    )


def export_trial(trial: str, size: int, min_veg_frac: float) -> Path:
    pairs = build_pairs(trial)
    if pairs.empty:
        raise SystemExit(f"No pairs for {trial}")

    cups = sorted(int(c) for c in pairs["cup_id"].unique())
    split_map = assign_cup_splits(cups)
    cup_to_split = {c: s for s, ids in split_map.items() for c in ids}

    ds = OUT_DIR / "dataset" / f"{trial}_controlnet_{size}"
    for split in ("train", "val", "test"):
        (ds / split / "conditioning").mkdir(parents=True, exist_ok=True)
        (ds / split / "target").mkdir(parents=True, exist_ok=True)

    rows_out = []
    dropped = 0
    for _, r in pairs.iterrows():
        split = cup_to_split[int(r["cup_id"])]
        src = PROJECT / r["crop_t"]
        tgt = PROJECT / r["crop_future"]
        if not src.exists() or not tgt.exists():
            dropped += 1
            continue

        img_t = rgba_to_rgb_white(Image.open(src))
        img_f = rgba_to_rgb_white(Image.open(tgt))
        crop_t, info_t = tight_square_crop(img_t)
        crop_f, info_f = tight_square_crop(img_f)

        # Keep all file-valid pairs so cup-level train/val/test splits stay intact.
        # Low-veg frames fall back to full-frame inside tight_square_crop.

        crop_t = crop_t.resize((size, size), Image.BICUBIC)
        crop_f = crop_f.resize((size, size), Image.BICUBIC)

        stem = f"cup{int(r['cup_id']):02d}_d{int(r['day_t']):02d}_to_d{int(r['day_future']):02d}"
        cond_path = ds / split / "conditioning" / f"{stem}.png"
        tgt_path = ds / split / "target" / f"{stem}.png"
        crop_t.save(cond_path)
        crop_f.save(tgt_path)

        meta = dict(r)
        meta.update(
            {
                "split": split,
                "prompt": climate_prompt(r),
                "conditioning_png": str(cond_path.relative_to(PROJECT)),
                "target_png": str(tgt_path.relative_to(PROJECT)),
                "veg_frac_t": info_t["veg_frac"],
                "veg_frac_future": info_f["veg_frac"],
                "used_full_frame_t": info_t["used_full_frame"],
                "used_full_frame_future": info_f["used_full_frame"],
                "size": size,
            }
        )
        rows_out.append(meta)

    meta_df = pd.DataFrame(rows_out)
    meta_df.to_csv(ds / "manifest.csv", index=False)

    # climate norm from train only (optional numeric conditioning later)
    clim_cols = [c for c in meta_df.columns if c.endswith("_now") or "mean4d" in c]
    train_df = meta_df[meta_df["split"] == "train"]
    stats = {
        c: {"mean": float(train_df[c].mean()), "std": float(train_df[c].std() or 1.0)} for c in clim_cols
    }
    (ds / "climate_norm.json").write_text(
        json.dumps(
            {
                "trial": trial,
                "size": size,
                "preprocess": "exgr_bbox_pad0.12_rgb_white",
                "min_veg_frac": min_veg_frac,
                "train_cups": split_map["train"],
                "val_cups": split_map["val"],
                "test_cups": split_map["test"],
                "n_dropped": dropped,
                "climate": stats,
            },
            indent=2,
        )
    )

    print(f"[{trial}] wrote {ds}  kept={len(meta_df)} dropped={dropped}")
    for split in ("train", "val", "test"):
        n = int((meta_df["split"] == split).sum())
        print(f"  {split}: {n}  cups={split_map[split]}")
    return ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", choices=["trial1", "trial2", "both"], default="both")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--min-veg-frac", type=float, default=0.005)
    args = parser.parse_args()

    trials = ["trial1", "trial2"] if args.trial == "both" else [args.trial]
    for t in trials:
        export_trial(t, args.size, args.min_veg_frac)


if __name__ == "__main__":
    main()

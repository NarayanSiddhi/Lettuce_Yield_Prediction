#!/usr/bin/env python3
"""
Trial 2 plant→tray check (future FW unknown per plant).

Setting (matches deploy when today weight is known at tray scale):
  - Today: tray-median FW is known on weigh-in days; per-plant FW is not.
  - Images: per-cup canopy area allocates today's tray weight across plants.
  - Forecast: Trial-1 AR-Δ (sensors + fw_t + prior gain) → each plant's FW @ t+4.
  - Score: mean(plant preds @ t+4) vs observed tray-median FW @ t+4.

Also reports a blind nowcast variant (image→fw_t, no tray today) as secondary.

Usage:
  python3 ml/eval_trial2_plant_mean_vs_tray.py --seeds 42,43,44,45,46
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

for k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(k, "1")

ML = Path(__file__).resolve().parent
PROJECT = ML.parent
OUT = ML / "future_fw_ar" / "trial2_plant_mean_vs_tray"
HORIZON = 4

AR_COLS = [
    "env_day_air_temp_c_mean_wmean",
    "env_night_air_temp_c_mean_wmean",
    "env_day_humidity_pct_mean_wmean",
    "env_night_humidity_pct_mean_wmean",
    "env_day_vpd_pa_mean_wmean",
    "env_night_vpd_pa_mean_wmean",
    "env_day_co2_ppm_mean_wmean",
    "env_act_day_light_duration_s_sum_wmean",
    "nutrient_ec_us_cm_mean_wmean",
    "exp_day_wmean",
    "fw_t",
    "gain_prev_4d",
]


def make_en(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                ElasticNet(
                    alpha=0.05, l1_ratio=0.25, random_state=seed, max_iter=50_000
                ),
            ),
        ]
    )


def load_t1_plant_pairs() -> pd.DataFrame:
    import runpy

    for cand in (
        ML / "future_fw_paper" / "plant_pairs_aim2.csv",
        ML / "aim2_phenotype_tuned" / "plant_pairs_aim2.csv",
    ):
        if cand.exists():
            df = pd.read_csv(cand)
            break
    else:
        api = runpy.run_path(str(ML / "train_aim2_phenotype_consistency.py"))
        df = api["build_pairs"]()
    df = df[np.isfinite(df["fw_future"].to_numpy(dtype=float))].copy()
    df["delta_4d"] = df["fw_future"] - df["fw_t"]
    return df.reset_index(drop=True)


def load_tray_climate(trial: str) -> pd.DataFrame:
    import runpy

    api = runpy.run_path(str(ML / "train_future_fw_autoregressive.py"))
    return api["build_tray_pairs"](trial)


def cup_masks(trial: str) -> pd.DataFrame:
    import runpy

    api = runpy.run_path(str(ML / "train_aim2_phenotype_consistency.py"))
    m = api["cup_mask_lookup"]()
    return m[m["trial"] == trial].copy()


def nearest_cup_areas(masks: pd.DataFrame, day: int, max_dist: int = 2) -> pd.DataFrame:
    """Per cup, area at DAT closest to day within max_dist."""
    rows = []
    for cid, g in masks.groupby("cup_id"):
        g = g.copy()
        g["dist"] = (g["DAT"] - day).abs()
        g = g[g["dist"] <= max_dist].sort_values(["dist", "DAT"])
        if g.empty:
            continue
        r = g.iloc[0]
        rows.append(
            {
                "cup_id": int(cid),
                "DAT_used": int(r["DAT"]),
                "dist": int(r["dist"]),
                "mask_area_px": float(r["mask_area_px"]),
                "plant_color_frac": float(r.get("plant_color_frac", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def train_nowcast_t1(seed: int) -> tuple[Pipeline, list[str]]:
    """Simple T1 nowcast: day + canopy → FW (for blind secondary route)."""
    plant = pd.read_csv(ML / "plant" / "plant_checkpoint_trial1_compact.csv")
    # prefer plant-level mask if present, else tray img mean
    feats = []
    for c in (
        "day",
        "plant_img_mask_area_px_last",
        "plant_img_mask_area_px_wmean",
        "img_mask_area_mean_wmean",
        "img_mask_area_sum_wmean",
        "plant_img_plant_color_frac_last",
        "exp_day_wmean",
    ):
        if c in plant.columns:
            feats.append(c)
    y = pd.to_numeric(plant["target_fw_g"], errors="coerce")
    X = plant[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    m = make_en(seed)
    m.fit(X[np.isfinite(y)], y[np.isfinite(y)])
    return m, feats


def nowcast_cups(
    model: Pipeline, feat_names: list[str], cups: pd.DataFrame, day: int
) -> np.ndarray:
    rows = []
    for _, r in cups.iterrows():
        d = {
            "day": float(day),
            "exp_day_wmean": float(day),
            "plant_img_mask_area_px_last": r["mask_area_px"],
            "plant_img_mask_area_px_wmean": r["mask_area_px"],
            "img_mask_area_mean_wmean": r["mask_area_px"],
            "img_mask_area_sum_wmean": r["mask_area_px"],
            "plant_img_plant_color_frac_last": r.get("plant_color_frac", np.nan),
        }
        rows.append([d.get(c, np.nan) for c in feat_names])
    X = np.asarray(rows, dtype=float)
    return np.clip(model.predict(X), 0.1, None)


def allocate_tray_by_canopy(tray_fw: float, cups: pd.DataFrame) -> np.ndarray:
    """
    Tray median FW is a *typical plant* weight, not total tray biomass.
    Scale each cup around that median by relative canopy size:
      fw_i = tray_median * (area_i / mean(area))
    so mean(fw_i) ≈ tray_median and bigger canopies get higher today-weight.
    """
    area = cups["mask_area_px"].to_numpy(dtype=float)
    area = np.where(np.isfinite(area) & (area > 0), area, np.nan)
    if not np.isfinite(area).any():
        return np.full(len(cups), tray_fw, dtype=float)
    area = np.where(np.isfinite(area), area, np.nanmean(area))
    rel = area / float(np.mean(area))
    return np.clip(tray_fw * rel, 0.1, None)


def fit_ar_delta(pairs: pd.DataFrame, seed: int) -> tuple[Pipeline, list[str]]:
    cols = [c for c in AR_COLS if c in pairs.columns]
    X = pairs[cols].to_numpy(dtype=float)
    y = pairs["delta_4d"].to_numpy(dtype=float)
    m = make_en(seed)
    m.fit(X, y)
    return m, cols


def climate_row(tray_pairs: pd.DataFrame, day_t: int) -> dict[str, float]:
    hit = tray_pairs[tray_pairs["day_t"] == day_t]
    if hit.empty:
        return {}
    r = hit.iloc[0]
    return {c: float(r[c]) for c in AR_COLS if c in r.index and c not in ("fw_t", "gain_prev_4d")}


def predict_plants(
    ar_model: Pipeline,
    ar_cols: list[str],
    clim: dict[str, float],
    fw_t: np.ndarray,
    gain: np.ndarray,
    day_t: int,
) -> np.ndarray:
    rows = []
    for fw, g in zip(fw_t, gain):
        d = {c: clim.get(c, np.nan) for c in ar_cols}
        d["fw_t"] = float(fw)
        d["gain_prev_4d"] = float(g) if np.isfinite(g) else np.nan
        d["exp_day_wmean"] = float(clim.get("exp_day_wmean", day_t))
        rows.append([d.get(c, np.nan) for c in ar_cols])
    X = np.asarray(rows, dtype=float)
    delta = ar_model.predict(X)
    return np.clip(fw_t + delta, 0.0, None)


def eval_seed(seed: int, max_dist: int = 2) -> tuple[pd.DataFrame, dict]:
    t1 = load_t1_plant_pairs()
    ar_model, ar_cols = fit_ar_delta(t1, seed)
    now_model, now_feats = train_nowcast_t1(seed)
    tray_pairs = load_tray_climate("trial2")
    masks = cup_masks("trial2")
    tray_fw = (
        pd.read_csv(ML / "checkpoint_trial2.csv")[["day", "target_median_fw_g"]]
        .drop_duplicates()
        .set_index("day")["target_median_fw_g"]
        .to_dict()
    )

    rows = []
    detail_rows = []
    for _, pr in tray_pairs.iterrows():
        day_t = int(pr["day_t"])
        day_f = int(pr["day_future"])
        if day_t not in tray_fw or day_f not in tray_fw:
            continue
        cups = nearest_cup_areas(masks, day_t, max_dist=max_dist)
        if cups.empty:
            continue
        # prior gain from canopy-share at t-4 if possible
        cups_prev = nearest_cup_areas(masks, day_t - HORIZON, max_dist=max_dist)
        clim = climate_row(tray_pairs, day_t)

        # --- A) today tray known → allocate by canopy ---
        fw_share = allocate_tray_by_canopy(float(tray_fw[day_t]), cups)
        if not cups_prev.empty:
            prev_map = cups_prev.set_index("cup_id")["mask_area_px"].to_dict()
            prev_tray = float(tray_fw.get(day_t - HORIZON, np.nan))
            if np.isfinite(prev_tray):
                tmp = cups.copy()
                tmp["mask_area_px"] = tmp["cup_id"].map(prev_map)
                fw_prev = allocate_tray_by_canopy(prev_tray, tmp.fillna({"mask_area_px": np.nanmean(fw_share)}))
                gain_share = fw_share - fw_prev
            else:
                gain_share = np.full(len(cups), np.nan)
        else:
            gain_share = np.full(len(cups), np.nan)

        pred_share = predict_plants(ar_model, ar_cols, clim, fw_share, gain_share, day_t)
        mean_share = float(np.mean(pred_share))

        # --- B) blind nowcast today ---
        fw_now = nowcast_cups(now_model, now_feats, cups, day_t)
        if not cups_prev.empty:
            # align prev cups to current cup list
            prev_map = {
                int(r.cup_id): float(r.mask_area_px) for r in cups_prev.itertuples()
            }
            cups_p = cups.copy()
            cups_p["mask_area_px"] = cups_p["cup_id"].map(prev_map)
            fw_now_prev = nowcast_cups(now_model, now_feats, cups_p.fillna(0), day_t - HORIZON)
            gain_now = fw_now - fw_now_prev
        else:
            gain_now = np.full(len(cups), np.nan)
        pred_now = predict_plants(ar_model, ar_cols, clim, fw_now, gain_now, day_t)
        mean_now = float(np.mean(pred_now))

        y = float(tray_fw[day_f])
        rows.append(
            {
                "seed": seed,
                "day_t": day_t,
                "day_future": day_f,
                "n_cups": int(len(cups)),
                "tray_fw_today": float(tray_fw[day_t]),
                "tray_fw_future_true": y,
                "mean_plant_pred_share": round(mean_share, 3),
                "mean_plant_pred_nowcast": round(mean_now, 3),
                "abs_err_share": abs(mean_share - y),
                "abs_err_nowcast": abs(mean_now - y),
                "persist_tray": float(tray_fw[day_t]),
                "abs_err_persist": abs(float(tray_fw[day_t]) - y),
            }
        )
        for i, cid in enumerate(cups["cup_id"].tolist()):
            detail_rows.append(
                {
                    "seed": seed,
                    "day_t": day_t,
                    "day_future": day_f,
                    "cup_id": cid,
                    "fw_t_share": float(fw_share[i]),
                    "fw_t_nowcast": float(fw_now[i]),
                    "pred_future_share": float(pred_share[i]),
                    "pred_future_nowcast": float(pred_now[i]),
                }
            )

    det = pd.DataFrame(detail_rows)
    summ = pd.DataFrame(rows)
    metrics = {
        "seed": seed,
        "n_intervals": int(len(summ)),
        "mae_share_vs_tray": float(summ["abs_err_share"].mean()) if len(summ) else float("nan"),
        "mae_nowcast_vs_tray": float(summ["abs_err_nowcast"].mean()) if len(summ) else float("nan"),
        "mae_persist_tray": float(summ["abs_err_persist"].mean()) if len(summ) else float("nan"),
        "mean_n_cups": float(summ["n_cups"].mean()) if len(summ) else float("nan"),
    }
    return summ, metrics, det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46")
    ap.add_argument("--max-dist", type=int, default=2)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_summ = []
    all_met = []
    all_det = []
    for seed in seeds:
        print(f"[seed {seed}] Trial2 plant-mean vs tray …", flush=True)
        summ, met, det = eval_seed(seed, max_dist=args.max_dist)
        print(
            f"  intervals={met['n_intervals']}  "
            f"MAE share→tray={met['mae_share_vs_tray']:.2f}  "
            f"MAE nowcast→tray={met['mae_nowcast_vs_tray']:.2f}  "
            f"persist={met['mae_persist_tray']:.2f}  "
            f"mean cups/day={met['mean_n_cups']:.1f}",
            flush=True,
        )
        all_summ.append(summ)
        all_met.append(met)
        all_det.append(det)

    summ = pd.concat(all_summ, ignore_index=True)
    met = pd.DataFrame(all_met)
    det = pd.concat(all_det, ignore_index=True)
    summ.to_csv(out / "per_interval.csv", index=False)
    met.to_csv(out / "per_seed_metrics.csv", index=False)
    det.to_csv(out / "per_plant_preds.csv", index=False)

    headline = {
        "protocol": (
            "Today tray median known; allocate to cups by canopy share; "
            "Trial1 AR-Δ per plant; mean plant pred vs tray median @ t+4."
        ),
        "secondary": "Blind image nowcast for fw_t (no tray today), then same AR.",
        "mae_share_vs_tray_mean_pm_sd": (
            f"{met['mae_share_vs_tray'].mean():.2f} ± {met['mae_share_vs_tray'].std(ddof=0):.2f}"
        ),
        "mae_nowcast_vs_tray_mean_pm_sd": (
            f"{met['mae_nowcast_vs_tray'].mean():.2f} ± {met['mae_nowcast_vs_tray'].std(ddof=0):.2f}"
        ),
        "mae_persist_tray_mean_pm_sd": (
            f"{met['mae_persist_tray'].mean():.2f} ± {met['mae_persist_tray'].std(ddof=0):.2f}"
        ),
        "n_intervals_per_seed": int(met["n_intervals"].iloc[0]) if len(met) else 0,
        "caveat": (
            "Per-plant truth on Trial 2 is unavailable; tray-median agreement is a "
            "proxy accuracy check. Cup images missing on some weigh-in days → nearest "
            f"DAT within ±{args.max_dist} d."
        ),
    }
    (out / "summary.json").write_text(json.dumps(headline, indent=2))
    print("\n=== HEADLINE ===", flush=True)
    print(json.dumps(headline, indent=2), flush=True)
    print(f"Wrote {out}/", flush=True)


if __name__ == "__main__":
    main()

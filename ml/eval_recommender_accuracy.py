#!/usr/bin/env python3
"""
Offline accuracy checks for the grower recommendation system.

We cannot run real greenhouse A/B tests here, so we report three layers:

  1) Forecast accuracy — status/predict vs observed FW @ t+4 (LODO by day)
  2) Days-to-target calibration — at day t, predict days until a later measured
     weight; compare to actual calendar days (Trial 1 plants only)
  3) Recommendation consistency (model-internal) — do reach_target tweaks move
     the *model* closer to the goal than doing nothing?
     (Not causal proof — labeled as consistency.)

Usage:
  python3 ml/eval_recommender_accuracy.py --seeds 42,43,44,45,46
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
OUT = ML / "recommendations" / "accuracy"
HORIZON = 4

# Import recommender helpers without running CLI
import runpy

REC = runpy.run_path(str(ML / "recommend_growth.py"))


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


def lodo_forecast_accuracy(pairs: pd.DataFrame, cols: list[str], seed: int) -> dict:
    """Same AR-Δ LODO the recommender uses for +4d predictions."""
    X = pairs[cols].to_numpy(dtype=float)
    y = pairs["fw_future"].to_numpy(dtype=float)
    fw_t = pairs["fw_t"].to_numpy(dtype=float)
    days = pairs["day_future"].to_numpy(dtype=int)
    y_d = y - fw_t
    pred = np.empty_like(y)
    for d in np.unique(days):
        te = days == d
        tr = ~te
        if tr.sum() < 5:
            pred[te] = fw_t[te] + (float(np.mean(y_d[tr])) if tr.any() else 0.0)
            continue
        m = make_en(seed)
        m.fit(X[tr], y_d[tr])
        pred[te] = np.clip(fw_t[te] + m.predict(X[te]), 0, None)
    mae = float(np.mean(np.abs(y - pred)))
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float("nan") if ss_tot < 1e-12 else 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot
    within_17 = float(np.mean(np.abs(y - pred) <= 17.0))
    within_mae = float(np.mean(np.abs(y - pred) <= mae))
    return {
        "mae_g": mae,
        "rmse_g": rmse,
        "r2": r2,
        "frac_within_17g": within_17,
        "frac_within_1x_mae": within_mae,
        "n": int(len(y)),
        "pred": pred,
        "y": y,
        "plant_id": pairs["plant_id"].to_numpy(dtype=int),
        "day_t": pairs["day_t"].to_numpy(dtype=int),
        "day_future": days,
        "fw_t": fw_t,
    }


def days_to_target_calibration(bundle, pairs: pd.DataFrame) -> pd.DataFrame:
    """
    For each plant, take earlier day_t and a later measured fw as target.
    Compare projected days-to-target vs actual (day_future - day_t).
    """
    rows = []
    # use pairs themselves: target = observed fw_future, ask "when from day_t?"
    for _, r in pairs.iterrows():
        pid = int(r["plant_id"])
        day_t = int(r["day_t"])
        target = float(r["fw_future"])
        actual_days = float(r["day_future"] - r["day_t"])
        try:
            out = REC["when_target"](bundle, pid, day_t, target)
        except SystemExit:
            continue
        tl = out.get("timeline_current_settings") or {}
        pred_days = tl.get("days_until_target")
        if pred_days is None:
            continue
        rows.append(
            {
                "plant_id": pid,
                "day_t": day_t,
                "target_fw_g": target,
                "actual_days": actual_days,
                "pred_days": float(pred_days),
                "abs_err_days": abs(float(pred_days) - actual_days),
                "pred_4d_fw": out.get("predicted_fw_in_4d_g"),
            }
        )
    return pd.DataFrame(rows)


def recommendation_consistency(bundle, pairs: pd.DataFrame, sample_n: int = 40) -> pd.DataFrame:
    """
    Model-internal: pick a goal above baseline (+15 g or actual future if higher),
    run reach_target, check if recommended pred is closer to goal than baseline.
    """
    rng = np.random.default_rng(0)
    idx = np.arange(len(pairs))
    if len(idx) > sample_n:
        idx = rng.choice(idx, size=sample_n, replace=False)
    rows = []
    for i in idx:
        r = pairs.iloc[int(i)]
        pid = int(r["plant_id"])
        day_t = int(r["day_t"])
        try:
            st = REC["status"](bundle, pid, day_t)
        except SystemExit:
            continue
        base = float(st["predicted_fw_future_g"])
        # ambitious but finite goal
        goal = max(base + 15.0, float(r["fw_future"]))
        try:
            rt = REC["reach_target"](bundle, pid, day_t, goal)
        except SystemExit:
            continue
        rec = float(rt["recommended_predicted_fw_g"])
        err_base = abs(base - goal)
        err_rec = abs(rec - goal)
        rows.append(
            {
                "plant_id": pid,
                "day_t": day_t,
                "goal_fw_g": goal,
                "baseline_pred_g": base,
                "recommended_pred_g": rec,
                "err_baseline": err_base,
                "err_recommended": err_rec,
                "improved": err_rec < err_base - 1e-6,
                "n_changes": len(rt.get("recommendations") or []),
                "feasibility": (rt.get("feasibility_for_4d") or {}).get("level"),
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46")
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = REC["load_pairs"]("trial1")
    cols = [c for c in REC["FORECAST_COLS"] if c in pairs.columns]

    # --- 1) Forecast LODO multi-seed ---
    forecast_rows = []
    pred_frames = []
    for seed in seeds:
        print(f"[seed {seed}] forecast LODO …", flush=True)
        met = lodo_forecast_accuracy(pairs, cols, seed)
        forecast_rows.append(
            {
                "seed": seed,
                "mae_g": met["mae_g"],
                "rmse_g": met["rmse_g"],
                "r2": met["r2"],
                "frac_within_17g": met["frac_within_17g"],
                "n": met["n"],
            }
        )
        pred_frames.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "plant_id": met["plant_id"],
                    "day_t": met["day_t"],
                    "day_future": met["day_future"],
                    "fw_t": met["fw_t"],
                    "fw_future_true": met["y"],
                    "fw_future_pred": met["pred"],
                    "abs_err": np.abs(met["y"] - met["pred"]),
                }
            )
        )
        print(
            f"  MAE={met['mae_g']:.2f}  R²={met['r2']:.3f}  "
            f"within±17g={100*met['frac_within_17g']:.1f}%",
            flush=True,
        )

    fdf = pd.DataFrame(forecast_rows)
    preds = pd.concat(pred_frames, ignore_index=True)
    fdf.to_csv(out / "forecast_lodo_per_seed.csv", index=False)
    preds.to_csv(out / "forecast_lodo_predictions.csv", index=False)

    # --- 2 & 3 use full-fit recommender bundle (seed 42) ---
    print("[bundle] build recommender + days-to-target + consistency …", flush=True)
    bundle = REC["build_bundle"](seed=seeds[0], trial="trial1")
    cal = days_to_target_calibration(bundle, pairs)
    cal.to_csv(out / "days_to_target_calibration.csv", index=False)
    cons = recommendation_consistency(bundle, pairs, sample_n=min(40, len(pairs)))
    cons.to_csv(out / "recommendation_consistency.csv", index=False)

    cal_mae = float(cal["abs_err_days"].mean()) if len(cal) else float("nan")
    cal_med = float(cal["abs_err_days"].median()) if len(cal) else float("nan")
    # for exact +4d targets, perfect would be 4.0
    improved_frac = float(cons["improved"].mean()) if len(cons) else float("nan")
    mean_err_drop = (
        float((cons["err_baseline"] - cons["err_recommended"]).mean()) if len(cons) else float("nan")
    )

    # Trial2 proxy from existing artifact if present
    t2_sum = ML / "future_fw_ar" / "trial2_plant_mean_vs_tray" / "summary.json"
    t2 = json.loads(t2_sum.read_text()) if t2_sum.exists() else {}

    headline = {
        "forecast_LODO": {
            "mae_g_mean_pm_sd": f"{fdf['mae_g'].mean():.2f} ± {fdf['mae_g'].std(ddof=0):.2f}",
            "r2_mean": float(fdf["r2"].mean()),
            "frac_within_17g_mean": float(fdf["frac_within_17g"].mean()),
            "n_pairs": int(fdf["n"].iloc[0]),
            "meaning": (
                "How accurate are status/predict +4d weight forecasts vs real weigh-ins "
                "(Trial 1, leave-one-harvest-day-out)."
            ),
        },
        "days_to_target_calibration": {
            "mae_days": cal_mae,
            "median_abs_err_days": cal_med,
            "n_cases": int(len(cal)),
            "meaning": (
                "At day t, ask when the plant will reach the weight later measured at t+4; "
                "compare predicted days vs actual 4 days. Measures timeline calibration."
            ),
        },
        "recommendation_consistency_model_internal": {
            "frac_improved_toward_goal": improved_frac,
            "mean_goal_error_drop_g": mean_err_drop,
            "n_cases": int(len(cons)),
            "meaning": (
                "Do suggested set-point tweaks move the *model prediction* closer to the "
                "goal than doing nothing? Consistency check — not a real greenhouse trial."
            ),
        },
        "trial2_proxy_plant_mean_vs_tray": {
            "mae_share_vs_tray": t2.get("mae_share_vs_tray_mean_pm_sd"),
            "mae_persist": t2.get("mae_persist_tray_mean_pm_sd"),
            "meaning": t2.get("protocol"),
        },
        "what_we_cannot_claim_yet": (
            "We have not run controlled experiments that apply recommended EC/temp and "
            "re-weigh plants. Causal accuracy of recommendations needs a real A/B or "
            "before/after trial."
        ),
    }
    (out / "recommender_accuracy_summary.json").write_text(json.dumps(headline, indent=2))

    print("\n=== RECOMMENDER ACCURACY SUMMARY ===", flush=True)
    print(json.dumps(headline, indent=2), flush=True)
    print(f"\nWrote {out}/", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Autoregressive future-FW routes (paper-grade comparison).

Professor note: learn day-to-day change; optionally step t→t+1→…→t+4.

Data constraint: fresh-weight labels exist only every 4 days. Therefore:

  A) direct_level   — X_t → FW_{t+4}           (current baseline)
  B) ar_delta_4d    — X_t → ΔFW over 4d; FW = fw_t + Δ   (label-honest AR)
  C) ar_daily_roll4 — learn daily ΔFW on log-interp FW between weigh-ins,
                      then roll t→t+1→…→t+4; evaluate only on true weigh-ins
                      (secondary / optimistic: mid-interval labels are synthetic)

Primary metric: LODO MAE (g) on true FW at t+4, multi-seed mean±SD.

Usage:
  python3 ml/train_future_fw_autoregressive.py --mode all --seeds 42,43,44,45,46
  # modes: trial1_plant | trial1_tray | trial2_tray | transfer_t1_to_t2 | all
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
OUT = ML / "future_fw_ar"
HORIZON = 4

PAIR_COLS = [
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

DAILY_FEAT = [
    "DAT",
    "T_air_C",
    "T_air_light_C",
    "RH_pct",
    "RH_light_pct",
    "CO2_ppm",
    "EC_uS_cm",
    "photoperiod_h",
    "DLI_mol_m2_d",
    "env_day_air_temp_c_mean",
    "env_night_air_temp_c_mean",
    "env_day_vpd_pa_mean",
    "env_day_co2_ppm_mean",
    "nutrient_ec_us_cm_mean",
    "fw_t",
    "gain_prev_1d",
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


def load_weighin_pairs_plant_trial1() -> pd.DataFrame:
    for cand in (
        ML / "future_fw_paper" / "plant_pairs_aim2.csv",
        ML / "aim2_phenotype_tuned" / "plant_pairs_aim2.csv",
    ):
        if cand.exists():
            df = pd.read_csv(cand)
            break
    else:
        import runpy

        api = runpy.run_path(str(ML / "train_aim2_phenotype_consistency.py"))
        df = api["build_pairs"]()
    df = df[np.isfinite(df["fw_future"].to_numpy(dtype=float))].copy()
    for c in PAIR_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["delta_4d"] = df["fw_future"] - df["fw_t"]
    df["trial"] = "trial1"
    df["unit"] = "plant"
    return df.reset_index(drop=True)


def load_weighin_pairs() -> pd.DataFrame:
    """Backward-compatible alias (Trial 1 plant pairs)."""
    return load_weighin_pairs_plant_trial1()


def _tray_climate_by_dat(trial: str) -> dict[int, dict[str, float]]:
    """
    Map DAT → climate features. Prefer checkpoint window means; if missing
    (Trial 2 sensors are often empty), fall back to daily drivers + daily sensors.
    """
    path = ML / f"checkpoint_{trial}.csv"
    ck = pd.read_csv(path).sort_values("day")
    out: dict[int, dict[str, float]] = {}
    for _, row in ck.iterrows():
        d = int(row["day"])
        feat = {}
        for c in PAIR_COLS:
            if c in ("fw_t", "gain_prev_4d"):
                continue
            if c in row.index and pd.notna(row[c]):
                feat[c] = float(row[c])
        out[d] = feat

    climate = load_daily_climate(trial)
    # map daily driver names → pair feature names
    rename = {
        "T_air_light_C": "env_day_air_temp_c_mean_wmean",
        "T_air_C": "env_night_air_temp_c_mean_wmean",  # approx if night missing
        "RH_light_pct": "env_day_humidity_pct_mean_wmean",
        "RH_pct": "env_night_humidity_pct_mean_wmean",
        "CO2_ppm": "env_day_co2_ppm_mean_wmean",
        "EC_uS_cm": "nutrient_ec_us_cm_mean_wmean",
        "DAT": "exp_day_wmean",
    }
    daily_direct = {
        "env_day_air_temp_c_mean": "env_day_air_temp_c_mean_wmean",
        "env_night_air_temp_c_mean": "env_night_air_temp_c_mean_wmean",
        "env_day_vpd_pa_mean": "env_day_vpd_pa_mean_wmean",
        "env_day_co2_ppm_mean": "env_day_co2_ppm_mean_wmean",
        "nutrient_ec_us_cm_mean": "nutrient_ec_us_cm_mean_wmean",
        "env_act_day_light_duration_s_sum": "env_act_day_light_duration_s_sum_wmean",
    }
    for _, r in climate.iterrows():
        d = int(r["DAT"])
        feat = out.get(d, {})
        for src, dst in daily_direct.items():
            if dst not in feat and src in r.index and pd.notna(r[src]):
                feat[dst] = float(r[src])
        for src, dst in rename.items():
            if dst not in feat and src in r.index and pd.notna(r[src]):
                feat[dst] = float(r[src])
        # VPD approx from drivers if still missing: leave NaN
        out[d] = feat
    return out


def build_tray_pairs(trial: str) -> pd.DataFrame:
    """Tray-median FW pairs (needed for Trial 2; no per-plant FW labels)."""
    path = ML / f"checkpoint_{trial}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    ck = pd.read_csv(path).sort_values("day")
    by = {int(r["day"]): r for _, r in ck.iterrows()}
    clim = _tray_climate_by_dat(trial)
    rows = []
    for day_t, row_t in by.items():
        day_f = day_t + HORIZON
        if day_f not in by:
            continue
        row_f = by[day_f]
        prev = day_t - HORIZON
        gain = (
            float(row_t["target_median_fw_g"]) - float(by[prev]["target_median_fw_g"])
            if prev in by
            else float("nan")
        )
        feat = {
            "trial": trial,
            "unit": "tray",
            "plant_id": 0,  # tray pseudo-id
            "day_t": day_t,
            "day_future": day_f,
            "fw_t": float(row_t["target_median_fw_g"]),
            "fw_future": float(row_f["target_median_fw_g"]),
            "gain_prev_4d": gain,
            "delta_4d": float(row_f["target_median_fw_g"]) - float(row_t["target_median_fw_g"]),
        }
        for c, v in clim.get(day_t, {}).items():
            feat[c] = v
        if "exp_day_wmean" not in feat or not np.isfinite(feat.get("exp_day_wmean", np.nan)):
            feat["exp_day_wmean"] = float(day_t)
        rows.append(feat)
    return pd.DataFrame(rows)


def load_daily_climate(trial: str = "trial1") -> pd.DataFrame:
    drivers = pd.read_csv(PROJECT / "Lettuce_model" / "data" / f"{trial}_daily_drivers.csv")
    drivers["date"] = pd.to_datetime(drivers["date"]).dt.normalize()
    drivers["EC_uS_cm"] = pd.to_numeric(drivers["EC_mS_cm"], errors="coerce") * 1000.0
    sens_path = PROJECT / f"sensor_features_daily_{trial}.csv"
    if sens_path.exists():
        sens = pd.read_csv(sens_path)
        sens["date"] = pd.to_datetime(sens["date"]).dt.normalize()
        keep_s = [
            c
            for c in (
                "date",
                "env_day_air_temp_c_mean",
                "env_night_air_temp_c_mean",
                "env_day_vpd_pa_mean",
                "env_day_co2_ppm_mean",
                "nutrient_ec_us_cm_mean",
                "env_act_day_light_duration_s_sum",
            )
            if c in sens.columns
        ]
        out = drivers.merge(sens[keep_s], on="date", how="left")
    else:
        out = drivers
    return out.sort_values("DAT").reset_index(drop=True)


def interpolate_plant_fw(pairs: pd.DataFrame, trial: str = "trial1") -> pd.DataFrame:
    """
    Log1p-linear daily FW between observed weigh-ins (per plant/tray unit).
    Mid-interval values are synthetic — used only to supervise daily Δ.
    """
    climate = load_daily_climate(trial)
    dat_min = int(climate["DAT"].min())
    dat_max = int(climate["DAT"].max())
    rows = []
    # true weigh-in map
    true = (
        pairs[["plant_id", "day_t", "fw_t"]]
        .drop_duplicates()
        .rename(columns={"day_t": "DAT", "fw_t": "fw_true"})
    )
    # also include terminal day_future as true points
    true2 = (
        pairs[["plant_id", "day_future", "fw_future"]]
        .drop_duplicates()
        .rename(columns={"day_future": "DAT", "fw_future": "fw_true"})
    )
    true_all = (
        pd.concat([true, true2], ignore_index=True)
        .dropna()
        .drop_duplicates(["plant_id", "DAT"])
        .sort_values(["plant_id", "DAT"])
    )

    for pid, g in true_all.groupby("plant_id"):
        obs_d = g["DAT"].to_numpy(dtype=float)
        obs_fw = g["fw_true"].to_numpy(dtype=float)
        if len(obs_d) < 2:
            continue
        all_d = np.arange(dat_min, dat_max + 1, dtype=int)
        # interpolate in log1p space, constant outside
        log_fw = np.interp(all_d, obs_d, np.log1p(np.clip(obs_fw, 0, None)))
        fw_i = np.expm1(log_fw)
        is_true = np.isin(all_d, obs_d.astype(int))
        for i, d in enumerate(all_d):
            rows.append(
                {
                    "plant_id": int(pid),
                    "DAT": int(d),
                    "fw_interp": float(fw_i[i]),
                    "fw_is_true": bool(is_true[i]),
                }
            )
    fw_daily = pd.DataFrame(rows)
    panel = fw_daily.merge(climate, on="DAT", how="left")
    panel = panel.sort_values(["plant_id", "DAT"]).reset_index(drop=True)
    panel["fw_next"] = panel.groupby("plant_id")["fw_interp"].shift(-1)
    panel["gain_prev_1d"] = panel.groupby("plant_id")["fw_interp"].diff()
    panel["delta_1d"] = panel["fw_next"] - panel["fw_interp"]
    panel["fw_t"] = panel["fw_interp"]
    return panel


def _mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def _rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def _r2(y, p) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float("nan") if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot


def lodo_direct_and_delta(pairs: pd.DataFrame, seed: int) -> dict:
    cols = [c for c in PAIR_COLS if c in pairs.columns]
    X = pairs[cols].to_numpy(dtype=float)
    y_fw = pairs["fw_future"].to_numpy(dtype=float)
    y_d = pairs["delta_4d"].to_numpy(dtype=float)
    fw_t = pairs["fw_t"].to_numpy(dtype=float)
    days = pairs["day_future"].to_numpy(dtype=int)

    pred_direct = np.empty_like(y_fw)
    pred_delta = np.empty_like(y_fw)
    model = make_en(seed)

    for d in np.unique(days):
        te = days == d
        tr = ~te
        if tr.sum() < 5:
            pred_direct[te] = fw_t[te] + float(np.mean(y_d[tr])) if tr.any() else fw_t[te]
            pred_delta[te] = pred_direct[te]
            continue
        m_lvl = clone(model)
        m_lvl.fit(X[tr], y_fw[tr])
        pred_direct[te] = m_lvl.predict(X[te])

        m_d = clone(model)
        m_d.fit(X[tr], y_d[tr])
        pred_delta[te] = fw_t[te] + m_d.predict(X[te])

    persist = fw_t
    return {
        "direct_level": {
            "mae_g": _mae(y_fw, pred_direct),
            "rmse_g": _rmse(y_fw, pred_direct),
            "r2": _r2(y_fw, pred_direct),
            "pred": pred_direct,
        },
        "ar_delta_4d": {
            "mae_g": _mae(y_fw, pred_delta),
            "rmse_g": _rmse(y_fw, pred_delta),
            "r2": _r2(y_fw, pred_delta),
            "pred": pred_delta,
        },
        "persist": {
            "mae_g": _mae(y_fw, persist),
            "rmse_g": _rmse(y_fw, persist),
            "r2": _r2(y_fw, persist),
            "pred": persist,
        },
        "y": y_fw,
        "days": days,
        "plant_id": pairs["plant_id"].to_numpy(dtype=int),
        "day_t": pairs["day_t"].to_numpy(dtype=int),
        "fw_t": fw_t,
    }


def roll_daily_ar(
    daily_model: Pipeline,
    feat_cols: list[str],
    panel: pd.DataFrame,
    plant_id: int,
    day_t: int,
    fw0: float,
    horizon: int = HORIZON,
) -> float:
    """Step t→t+1→…→t+horizon starting from true fw0 at day_t."""
    fw = float(fw0)
    gain = 0.0
    # initial gain: previous true/interp day if available
    prev = panel[(panel["plant_id"] == plant_id) & (panel["DAT"] == day_t - 1)]
    if len(prev):
        gain = float(fw0 - float(prev.iloc[0]["fw_interp"]))
    for step in range(horizon):
        d = day_t + step
        row = panel[(panel["plant_id"] == plant_id) & (panel["DAT"] == d)]
        if row.empty:
            # climate-only fallback from any plant on that DAT
            row = panel[panel["DAT"] == d].head(1)
        if row.empty:
            break
        r = row.iloc[0].copy()
        r["fw_t"] = fw
        r["gain_prev_1d"] = gain
        r["DAT"] = float(d)
        x = r[feat_cols].to_numpy(dtype=float).reshape(1, -1)
        dlt = float(daily_model.predict(x)[0])
        # keep deltas physically mild
        dlt = float(np.clip(dlt, -20.0, 80.0))
        fw_next = max(0.0, fw + dlt)
        gain = fw_next - fw
        fw = fw_next
    return fw


def lodo_daily_roll(pairs: pd.DataFrame, panel: pd.DataFrame, seed: int) -> dict:
    """
    LODO by held-out weigh-in day_future.
    Train daily Δ on interpolated steps that do not land on a test weigh-in day.
    Evaluate by rolling 4 daily steps from each test pair's day_t.
    """
    feat_cols = [c for c in DAILY_FEAT if c in panel.columns]
    y_fw = pairs["fw_future"].to_numpy(dtype=float)
    days_f = pairs["day_future"].to_numpy(dtype=int)
    pred = np.empty_like(y_fw)
    model = make_en(seed)

    for d_te in np.unique(days_f):
        te = days_f == d_te
        # train daily rows: next-day target not equal to held-out weigh-in day
        # also drop steps inside (d_te-HORIZON, d_te] to reduce leakage from that interval
        tr_mask = (
            panel["delta_1d"].notna()
            & (panel["DAT"] + 1 != d_te)
            & ~((panel["DAT"] >= d_te - HORIZON) & (panel["DAT"] < d_te))
        )
        tr = panel.loc[tr_mask]
        if len(tr) < 20:
            # fallback: all non-test-landing steps
            tr = panel[panel["delta_1d"].notna() & (panel["DAT"] + 1 != d_te)]
        X_tr = tr[feat_cols].to_numpy(dtype=float)
        y_tr = tr["delta_1d"].to_numpy(dtype=float)
        m = clone(model)
        m.fit(X_tr, y_tr)

        for i in np.where(te)[0]:
            pid = int(pairs.iloc[i]["plant_id"])
            day_t = int(pairs.iloc[i]["day_t"])
            fw0 = float(pairs.iloc[i]["fw_t"])
            pred[i] = roll_daily_ar(m, feat_cols, panel, pid, day_t, fw0, HORIZON)

    return {
        "ar_daily_roll4": {
            "mae_g": _mae(y_fw, pred),
            "rmse_g": _rmse(y_fw, pred),
            "r2": _r2(y_fw, pred),
            "pred": pred,
        }
    }


def fit_full_models(pairs: pd.DataFrame, panel: pd.DataFrame, seed: int) -> dict:
    """Fit on all data for recommender export (explanatory / deployment artifacts)."""
    cols = [c for c in PAIR_COLS if c in pairs.columns]
    X = pairs[cols].to_numpy(dtype=float)
    m_direct = make_en(seed)
    m_direct.fit(X, pairs["fw_future"].to_numpy(dtype=float))
    m_delta = make_en(seed)
    m_delta.fit(X, pairs["delta_4d"].to_numpy(dtype=float))

    feat_cols = [c for c in DAILY_FEAT if c in panel.columns]
    tr = panel[panel["delta_1d"].notna()]
    m_daily = make_en(seed)
    m_daily.fit(tr[feat_cols].to_numpy(dtype=float), tr["delta_1d"].to_numpy(dtype=float))
    return {
        "pair_cols": cols,
        "daily_feat_cols": feat_cols,
        "direct": m_direct,
        "delta4": m_delta,
        "daily": m_daily,
    }


def mean_sd_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    g = (
        df.groupby("route", as_index=False)
        .agg(
            mae_mean=("mae_g", "mean"),
            mae_sd=("mae_g", "std"),
            rmse_mean=("rmse_g", "mean"),
            rmse_sd=("rmse_g", "std"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            n_seeds=("seed", "count"),
        )
        .sort_values("mae_mean")
    )
    g["mae_mean_pm_sd"] = g.apply(
        lambda r: f"{r['mae_mean']:.2f} ± {0.0 if pd.isna(r['mae_sd']) else r['mae_sd']:.2f}",
        axis=1,
    )
    return g


def run_lodo_setting(
    pairs: pd.DataFrame,
    trial: str,
    unit: str,
    seeds: list[int],
    out: Path,
    save_models: bool = False,
) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "pairs.csv", index=False)
    panel = interpolate_plant_fw(pairs, trial=trial)
    panel.to_csv(out / "daily_fw_interp_panel.csv", index=False)
    print(
        f"[{trial}/{unit}] pairs={len(pairs)} daily_panel={len(panel)}",
        flush=True,
    )

    per_seed = []
    pred_frames = []
    for seed in seeds:
        print(f"  [seed {seed}] LODO …", flush=True)
        a = lodo_direct_and_delta(pairs, seed)
        b = lodo_daily_roll(pairs, panel, seed)
        routes = {**{k: a[k] for k in ("direct_level", "ar_delta_4d", "persist")}, **b}
        for route, met in routes.items():
            per_seed.append(
                {
                    "trial": trial,
                    "unit": unit,
                    "seed": seed,
                    "route": route,
                    "mae_g": met["mae_g"],
                    "rmse_g": met["rmse_g"],
                    "r2": met["r2"],
                    "n_pairs": int(len(a["y"])),
                    "split": "LODO",
                    "note": (
                        "secondary_interp_labels"
                        if route == "ar_daily_roll4"
                        else "true_weighin_labels"
                    ),
                }
            )
            print(
                f"    {route:16s}  MAE={met['mae_g']:.2f}  R²={met['r2']:.3f}",
                flush=True,
            )
        pred_frames.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "trial": trial,
                    "unit": unit,
                    "plant_id": a["plant_id"],
                    "day_t": a["day_t"],
                    "day_future": a["days"],
                    "fw_t": a["fw_t"],
                    "fw_future": a["y"],
                    "pred_persist": a["persist"]["pred"],
                    "pred_direct_level": a["direct_level"]["pred"],
                    "pred_ar_delta_4d": a["ar_delta_4d"]["pred"],
                    "pred_ar_daily_roll4": b["ar_daily_roll4"]["pred"],
                }
            )
        )

    per = pd.DataFrame(per_seed)
    summary = mean_sd_table(per_seed)
    summary.insert(0, "unit", unit)
    summary.insert(0, "trial", trial)
    per.to_csv(out / "results_ar_per_seed.csv", index=False)
    summary.to_csv(out / "results_ar_mean_sd.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(
        out / "predictions_ar_lodo.csv", index=False
    )

    if save_models:
        import joblib

        fitted = fit_full_models(pairs, panel, seed=seeds[0])
        joblib.dump(
            {
                "trial": trial,
                "unit": unit,
                "horizon": HORIZON,
                "pair_cols": fitted["pair_cols"],
                "daily_feat_cols": fitted["daily_feat_cols"],
                "direct": fitted["direct"],
                "delta4": fitted["delta4"],
                "daily": fitted["daily"],
                "panel": panel,
                "pairs": pairs,
                "summary": summary.to_dict(orient="records"),
            },
            out / "ar_models_seed42.joblib",
        )
        # keep recommender default path for trial1 plant
        if trial == "trial1" and unit == "plant":
            joblib.dump(
                joblib.load(out / "ar_models_seed42.joblib"),
                OUT / "ar_models_seed42.joblib",
            )

    best = summary.iloc[0]
    meta = {
        "trial": trial,
        "unit": unit,
        "horizon_days": HORIZON,
        "seeds": seeds,
        "n_pairs": int(len(pairs)),
        "split": "LODO",
        "headline_best_route": best["route"],
        "headline_mae": best["mae_mean_pm_sd"],
        "table": summary.to_dict(orient="records"),
        "caveat": (
            "Tray LODO has only ~7 pairs — report with caution. "
            "Daily AR uses interpolated mid-weigh-in FW (secondary)."
            if unit == "tray"
            else "Daily AR uses interpolated mid-weigh-in FW (secondary)."
        ),
    }
    (out / "ar_summary.json").write_text(json.dumps(meta, indent=2))
    print(
        f"  HEADLINE [{trial}/{unit}]: {best['route']}  {best['mae_mean_pm_sd']}",
        flush=True,
    )
    return summary


def run_transfer_t1_to_t2(seeds: list[int], out: Path) -> pd.DataFrame:
    """Train on Trial 1 plant pairs; test on Trial 2 tray pairs (hard transfer)."""
    out.mkdir(parents=True, exist_ok=True)
    tr = load_weighin_pairs_plant_trial1()
    te = build_tray_pairs("trial2")
    tr.to_csv(out / "train_pairs_trial1_plant.csv", index=False)
    te.to_csv(out / "test_pairs_trial2_tray.csv", index=False)
    cols = [c for c in PAIR_COLS if c in tr.columns and c in te.columns]
    Xtr = tr[cols].to_numpy(dtype=float)
    ytr_fw = tr["fw_future"].to_numpy(dtype=float)
    ytr_d = tr["delta_4d"].to_numpy(dtype=float)
    Xte = te[cols].to_numpy(dtype=float)
    yte = te["fw_future"].to_numpy(dtype=float)
    fw_te = te["fw_t"].to_numpy(dtype=float)

    rows = []
    pred_frames = []
    for seed in seeds:
        m_lvl = make_en(seed)
        m_lvl.fit(Xtr, ytr_fw)
        p_lvl = m_lvl.predict(Xte)
        m_d = make_en(seed)
        m_d.fit(Xtr, ytr_d)
        p_d = fw_te + m_d.predict(Xte)
        p_pers = fw_te
        for route, pred in [
            ("persist", p_pers),
            ("direct_level", p_lvl),
            ("ar_delta_4d", p_d),
        ]:
            rows.append(
                {
                    "trial": "trial2",
                    "unit": "tray",
                    "seed": seed,
                    "route": route,
                    "mae_g": _mae(yte, pred),
                    "rmse_g": _rmse(yte, pred),
                    "r2": _r2(yte, pred),
                    "n_pairs": int(len(yte)),
                    "split": "train_T1_plant→test_T2_tray",
                    "note": "transfer",
                }
            )
        pred_frames.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "day_t": te["day_t"],
                    "day_future": te["day_future"],
                    "fw_t": fw_te,
                    "fw_future": yte,
                    "pred_persist": p_pers,
                    "pred_direct_level": p_lvl,
                    "pred_ar_delta_4d": p_d,
                }
            )
        )
        print(
            f"  [transfer seed {seed}] ar_delta MAE={_mae(yte, p_d):.2f}  "
            f"direct MAE={_mae(yte, p_lvl):.2f}  persist={_mae(yte, p_pers):.2f}",
            flush=True,
        )

    per = pd.DataFrame(rows)
    summary = mean_sd_table(rows)
    summary.insert(0, "unit", "tray")
    summary.insert(0, "trial", "trial2")
    summary.insert(0, "split", "train_T1_plant→test_T2_tray")
    per.to_csv(out / "results_transfer_per_seed.csv", index=False)
    summary.to_csv(out / "results_transfer_mean_sd.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(
        out / "predictions_transfer.csv", index=False
    )
    (out / "transfer_summary.json").write_text(
        json.dumps(
            {
                "split": "train_T1_plant→test_T2_tray",
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "table": summary.to_dict(orient="records"),
                "caveat": (
                    "Trial 2 has tray-median FW only (no per-plant labels). "
                    "This transfer test is the primary Trial-2 generalization check."
                ),
            },
            indent=2,
        )
    )
    print("  TRANSFER HEADLINE:", flush=True)
    print(summary[["route", "mae_mean_pm_sd", "r2_mean"]].to_string(index=False), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=[
            "trial1_plant",
            "trial1_tray",
            "trial2_tray",
            "transfer_t1_to_t2",
            "all",
        ],
    )
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46")
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    modes = (
        ["trial1_plant", "trial1_tray", "trial2_tray", "transfer_t1_to_t2"]
        if args.mode == "all"
        else [args.mode]
    )
    all_sum = []
    for mode in modes:
        if mode == "trial1_plant":
            pairs = load_weighin_pairs_plant_trial1()
            s = run_lodo_setting(
                pairs, "trial1", "plant", seeds, root / "trial1_plant", save_models=True
            )
            # also mirror summary at root for backward compatibility
            s.to_csv(root / "results_ar_mean_sd.csv", index=False)
            all_sum.append(s)
        elif mode == "trial1_tray":
            pairs = build_tray_pairs("trial1")
            all_sum.append(
                run_lodo_setting(pairs, "trial1", "tray", seeds, root / "trial1_tray")
            )
        elif mode == "trial2_tray":
            pairs = build_tray_pairs("trial2")
            all_sum.append(
                run_lodo_setting(
                    pairs, "trial2", "tray", seeds, root / "trial2_tray", save_models=True
                )
            )
        elif mode == "transfer_t1_to_t2":
            all_sum.append(run_transfer_t1_to_t2(seeds, root / "transfer_t1_to_t2"))

    if all_sum:
        combo = pd.concat(all_sum, ignore_index=True)
        combo.to_csv(root / "results_ar_all_settings.csv", index=False)
        (root / "ar_all_summary.json").write_text(
            json.dumps(
                {
                    "modes": modes,
                    "seeds": seeds,
                    "tables": combo.to_dict(orient="records"),
                },
                indent=2,
            )
        )
    print(f"\nWrote {root}/", flush=True)


if __name__ == "__main__":
    main()

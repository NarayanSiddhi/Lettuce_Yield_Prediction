#!/usr/bin/env python3
"""
Grower / greenhouse recommendation layer (paper-grade decision support).

Built on Trial 1 future-FW autoregressive change models:
  - primary +4d: predict ΔFW then FW = fw_t + Δ  (ar_delta_4d)
  - timeline: daily Δ roll t→t+1→… when artifacts exist (ar_daily_roll)

Questions this answers (CLI / JSON):
  1. status     — weight after N days (--horizon or --until-day) + drivers
  2. what_if    — if I change EC / temp / CO2 / …, how does predicted FW move?
  3. predict    — same future-weight ask; user chooses horizon / target day
  4. reach_target — goal weight: set-point tweaks for +4d + estimated days-to-target
  5. when       — only: after how many days ≈ target weight under current settings
  6. report     — batch recommendations for all plants on a chosen day

Explainability:
  Local SHAP (LinearExplainer) + template NLP summaries grounded in those numbers.
  NLP improves readability of the Summary line; it does not shrink forecast error.

Literature stance:
  Take: FGTD/Lin multimodal forecast + attribution for management hypotheses.
  Skip: closed-loop RL / claiming causal deficiency diagnosis.
  Improve: constrained counterfactuals + LODO error bands + SHAP/NLP explanations.

Usage:
  python3 ml/recommend_growth.py status --plant-id 3 --day 16
  python3 ml/recommend_growth.py what_if --plant-id 3 --day 16 --set nutrient_ec_us_cm_mean_wmean=1800
  python3 ml/recommend_growth.py reach_target --plant-id 3 --day 16 --target-fw 120
  python3 ml/recommend_growth.py report --day 16
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
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
OUT = ML / "recommendations"
HORIZON = 4

FORECAST_COLS = [
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

# Knobs a grower can actually try to change (not fw_t / gain / DAT)
ACTIONABLE = [
    "nutrient_ec_us_cm_mean_wmean",
    "env_day_co2_ppm_mean_wmean",
    "env_day_air_temp_c_mean_wmean",
    "env_night_air_temp_c_mean_wmean",
    "env_day_vpd_pa_mean_wmean",
    "env_night_vpd_pa_mean_wmean",
    "env_day_humidity_pct_mean_wmean",
    "env_night_humidity_pct_mean_wmean",
    "env_act_day_light_duration_s_sum_wmean",
]

FRIENDLY = {
    "nutrient_ec_us_cm_mean_wmean": "Nutrient EC (µS/cm)",
    "env_day_co2_ppm_mean_wmean": "Daytime CO₂ (ppm)",
    "env_day_air_temp_c_mean_wmean": "Day air temperature (°C)",
    "env_night_air_temp_c_mean_wmean": "Night air temperature (°C)",
    "env_day_vpd_pa_mean_wmean": "Day VPD (Pa)",
    "env_night_vpd_pa_mean_wmean": "Night VPD (Pa)",
    "env_day_humidity_pct_mean_wmean": "Day humidity (%)",
    "env_night_humidity_pct_mean_wmean": "Night humidity (%)",
    "env_act_day_light_duration_s_sum_wmean": "Day light-on duration (s, window)",
    "fw_t": "Current fresh weight (g)",
    "gain_prev_4d": "Prior 4-day weight gain (g)",
    "exp_day_wmean": "Days into grow cycle",
}

DISCLAIMER = (
    "Decision-support only: recommendations are model counterfactuals within the "
    "observed Trial 1 range, not proven causal effects or automatic greenhouse control. "
    "Validate with controlled experiments before operational changes."
)


def load_pairs(trial: str = "trial1") -> pd.DataFrame:
    """
    Trial 1: per-plant pairs (primary recommender).
    Trial 2: tray-median pairs (plant_id=0) from AR artifacts / builder.
    """
    if trial == "trial2":
        cand = ML / "future_fw_ar" / "trial2_tray" / "pairs.csv"
        if cand.exists():
            df = pd.read_csv(cand)
        else:
            import runpy

            api = runpy.run_path(str(ML / "train_future_fw_autoregressive.py"))
            df = api["build_tray_pairs"]("trial2")
    else:
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
    for c in FORECAST_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def make_model(seed: int = 42) -> Pipeline:
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


@dataclass
class ModelBundle:
    model: object  # predicts ΔFW over HORIZON days (ar_delta_4d)
    cols: list[str]
    pairs: pd.DataFrame
    lodo_mae: float
    ranges: dict[str, dict[str, float]]
    importance: list[dict]
    shap_explainer: object | None = None
    shap_method: str = "none"
    route: str = "ar_delta_4d"
    daily_model: object | None = None
    daily_feat_cols: list[str] | None = None
    daily_panel: pd.DataFrame | None = None


def lodo_mae_delta(
    model, X: np.ndarray, y_fw: np.ndarray, fw_t: np.ndarray, days: np.ndarray
) -> float:
    """LODO MAE for AR-Δ: predict Δ, reconstruct FW = fw_t + Δ."""
    y_d = y_fw - fw_t
    pred = np.empty_like(y_fw, dtype=float)
    for d in np.unique(days):
        te = days == d
        tr = ~te
        if tr.sum() < 5:
            pred[te] = fw_t[te] + (float(np.mean(y_d[tr])) if tr.any() else 0.0)
            continue
        m = clone(model)
        m.fit(X[tr], y_d[tr])
        pred[te] = fw_t[te] + m.predict(X[te])
    return float(np.mean(np.abs(y_fw - pred)))


def feature_ranges(df: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        out[c] = {
            "min": float(s.min()),
            "p05": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
        }
    return out


def simple_importance(model, X: np.ndarray, y: np.ndarray, cols: list[str]) -> list[dict]:
    """|coef| after fit on all data (ElasticNet) — fast driver list for UI."""
    m = clone(model)
    m.fit(X, y)
    coefs = m.named_steps["model"].coef_
    rows = []
    for c, w in zip(cols, coefs):
        rows.append(
            {
                "feature": c,
                "name": FRIENDLY.get(c, c),
                "coef": float(w),
                "abs_coef": float(abs(w)),
                "actionable": c in ACTIONABLE,
            }
        )
    rows.sort(key=lambda r: -r["abs_coef"])
    return rows


def _build_shap_explainer(fitted: Pipeline, X: np.ndarray) -> tuple[object | None, str]:
    """LinearExplainer on imputed+scaled features; None if shap unavailable."""
    try:
        import shap
    except Exception:
        return None, "none"
    try:
        Xi = fitted.named_steps["impute"].transform(X)
        Xs = fitted.named_steps["scale"].transform(Xi)
        explainer = shap.LinearExplainer(fitted.named_steps["model"], Xs)
        return explainer, "LinearExplainer"
    except Exception:
        return None, "none"


def _try_load_daily_ar(
    trial: str = "trial1", seed: int = 42
) -> tuple[object | None, list[str] | None, pd.DataFrame | None]:
    """Load daily AR roll artifacts from ml/future_fw_ar/ if present."""
    cands = []
    if trial == "trial2":
        cands.append(ML / "future_fw_ar" / "trial2_tray" / "ar_models_seed42.joblib")
    cands.extend(
        [
            ML / "future_fw_ar" / "trial1_plant" / "ar_models_seed42.joblib",
            ML / "future_fw_ar" / "ar_models_seed42.joblib",
        ]
    )
    try:
        import joblib
    except Exception:
        return None, None, None
    for path in cands:
        if not path.exists():
            continue
        try:
            blob = joblib.load(path)
            return blob.get("daily"), blob.get("daily_feat_cols"), blob.get("panel")
        except Exception:
            continue
    return None, None, None


def build_bundle(seed: int = 42, trial: str = "trial1") -> ModelBundle:
    pairs = load_pairs(trial=trial)
    cols = [c for c in FORECAST_COLS if c in pairs.columns]
    X = pairs[cols].to_numpy(dtype=float)
    y = pairs["fw_future"].to_numpy(dtype=float)
    fw_t = pairs["fw_t"].to_numpy(dtype=float)
    days = pairs["day_future"].to_numpy(dtype=int)
    y_delta = y - fw_t
    model = make_model(seed)
    mae = lodo_mae_delta(model, X, y, fw_t, days)
    # fit full AR-Δ for counterfactuals / SHAP (explanatory use)
    fitted = clone(model)
    fitted.fit(X, y_delta)
    shap_explainer, shap_method = _build_shap_explainer(fitted, X)
    daily_model, daily_cols, daily_panel = _try_load_daily_ar(trial=trial, seed=seed)
    return ModelBundle(
        model=fitted,
        cols=cols,
        pairs=pairs,
        lodo_mae=mae,
        ranges=feature_ranges(pairs, cols),
        importance=simple_importance(model, X, y_delta, cols),
        shap_explainer=shap_explainer,
        shap_method=shap_method,
        route="ar_delta_4d",
        daily_model=daily_model,
        daily_feat_cols=daily_cols,
        daily_panel=daily_panel,
    )


def local_shap(bundle: ModelBundle, x: np.ndarray, top_k: int = 6) -> list[dict]:
    """
    Per-prediction feature contributions in grams.
    Prefers SHAP; falls back to coef × standardized deviation from training mean.
    """
    pipe = bundle.model
    Xi = pipe.named_steps["impute"].transform(x)
    Xs = pipe.named_steps["scale"].transform(Xi)
    method = bundle.shap_method
    if bundle.shap_explainer is not None:
        try:
            sv = np.asarray(bundle.shap_explainer.shap_values(Xs), dtype=float).reshape(-1)
            method = bundle.shap_method
        except Exception:
            sv = None
    else:
        sv = None
    if sv is None:
        # Fallback: linear contribution in scaled space (same spirit as SHAP for EN)
        coefs = pipe.named_steps["model"].coef_
        sv = (coefs * Xs[0]).astype(float)
        method = "coef_x_zscore_fallback"

    rows = []
    for c, s, raw in zip(bundle.cols, sv, x.reshape(-1)):
        rows.append(
            {
                "feature": c,
                "name": FRIENDLY.get(c, c),
                "shap_g": round(float(s), 2),
                "value": None if not np.isfinite(raw) else round(float(raw), 3),
                "actionable": c in ACTIONABLE,
                "pushes": "up" if float(s) > 0 else "down",
            }
        )
    rows.sort(key=lambda r: -abs(r["shap_g"]))
    for r in rows:
        r["method"] = method
    return rows[:top_k]


# ---------------------------------------------------------------------------
# Template NLP (grounded in model numbers + SHAP) — clearer wording, not
# higher predictive accuracy.
# ---------------------------------------------------------------------------

def _join_natural(parts: list[str]) -> str:
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def nlg_why_from_shap(shap_rows: list[dict], top_k: int = 3, min_abs: float = 0.8) -> str:
    """Plain-English plant factors from local SHAP (or fallback) contributions."""
    bits = []
    for r in shap_rows[:top_k]:
        if abs(r["shap_g"]) < min_abs:
            continue
        name = r["name"]
        mag = abs(r["shap_g"])
        if r["pushes"] == "up":
            verb = "is helping growth"
        else:
            verb = "is slowing growth"
        knob = " (you can adjust this)" if r.get("actionable") else ""
        bits.append(f"{name} {verb} by about {mag:.1f} g{knob}")
    if not bits:
        return "No single factor stands out strongly for this plant right now."
    return _join_natural(bits) + "."


def nlg_status(
    plant_id: int,
    day_t: int,
    fw_now: float,
    pred: float,
    mae: float,
    shap_rows: list[dict],
) -> str:
    why = nlg_why_from_shap(shap_rows, top_k=3)
    return (
        f"Plant {plant_id} weighs {fw_now:.1f} g today (day {day_t}). "
        f"In about {HORIZON} days it is expected to be around {pred:.1f} g. "
        f"Main reasons: {why}"
    )


def nlg_what_if(
    plant_id: int,
    day_t: int,
    base: float,
    new: float,
    mae: float,
    changes: list[dict],
    shap_before: list[dict],
    shap_after: list[dict],
) -> str:
    delta = new - base
    direction = "higher" if delta > 0 else ("lower" if delta < 0 else "about the same")
    change_bits = [
        f"{c['name']} from {c['from']:.2f} to {c['to']:.2f}" for c in changes
    ]
    before_map = {r["feature"]: r["shap_g"] for r in shap_before}
    moved = []
    for r in shap_after:
        if r["feature"] not in before_map:
            continue
        d = r["shap_g"] - before_map[r["feature"]]
        if abs(d) >= 0.5 and r.get("actionable"):
            if d > 0:
                moved.append(f"{r['name']} would help more (+{d:.1f} g)")
            else:
                moved.append(f"{r['name']} would help less ({d:.1f} g)")
    moved = moved[:2]
    why_now = nlg_why_from_shap(shap_after, top_k=2, min_abs=0.5)
    return (
        f"For plant {plant_id} on day {day_t}, changing "
        f"{_join_natural(change_bits)} moves the expected weight in {HORIZON} days "
        f"from {base:.1f} g to {new:.1f} g ({direction} by {abs(delta):.1f} g). "
        + (f"Biggest setting effects: {_join_natural(moved)}. " if moved else "")
        + f"After that change: {why_now}"
    )


def nlg_when(
    plant_id: int,
    day_t: int,
    fw_now: float,
    pred_4d: float,
    target_fw: float,
    feas: dict,
    timeline: dict,
    shap_rows: list[dict],
) -> str:
    days = timeline.get("days_until_target")
    why = nlg_why_from_shap(shap_rows, top_k=2, min_abs=0.8)
    if days is None:
        when_bit = (
            f"At current settings it may not reach {target_fw:.0f} g "
            f"before late in the grow cycle."
        )
    else:
        when_bit = (
            f"At current settings it looks closer to about {days:.1f} days "
            f"from today (around day {timeline.get('reach_on_approx_dat')}), "
            f"not necessarily in a single {HORIZON}-day jump."
        )
    # Soften feasibility message: drop "model" wording if present
    feas_msg = (feas.get("message") or "").replace("A model that still 'reaches' the target in 4d by cranking set-points is likely over-confident — use the timeline (days-to-target) instead.", "Prefer the longer timeline over assuming a single 4-day jump.")
    return (
        f"Plant {plant_id} is {fw_now:.1f} g today (day {day_t}). "
        f"In the next {HORIZON} days alone it is expected around {pred_4d:.1f} g. "
        f"{feas_msg} "
        f"{when_bit} "
        f"Main reasons: {why}"
    )


def nlg_reach_target(
    plant_id: int,
    day_t: int,
    fw_now: float,
    target_fw: float,
    base: float,
    rec_pred: float,
    feas: dict,
    change_list: list[dict],
    timeline_cur: dict,
    timeline_rec: dict,
    shap_rows: list[dict],
    mae: float,
) -> str:
    why = nlg_why_from_shap(shap_rows, top_k=2, min_abs=0.8)
    days_cur = timeline_cur.get("days_until_target")
    days_rec = timeline_rec.get("days_until_target")
    if change_list:
        tweaks = _join_natural(
            [f"{c['name']} {c['from']:.2f}→{c['to']:.2f}" for c in change_list]
        )
        tweak_bit = (
            f"For the next weigh-in, try {tweaks}; "
            f"expected weight then moves from about {base:.1f} g to about {rec_pred:.1f} g."
        )
    else:
        tweak_bit = (
            f"No helpful in-range setting change stood out for the next {HORIZON} days."
        )
    if days_cur is not None:
        time_bit = (
            f"At current settings, plan on roughly {days_cur:.1f} days to reach {target_fw:.0f} g"
            + (
                f", or about {days_rec:.1f} days if you hold the suggested settings"
                if days_rec is not None and change_list
                else ""
            )
            + "."
        )
    else:
        time_bit = (
            f"Even at current settings, {target_fw:.0f} g may sit outside the late grow window."
        )
    caution = ""
    if feas.get("level") in ("optimistic", "unlikely_in_4d"):
        caution = (
            f" Reaching {target_fw:.0f} g in only {HORIZON} days looks "
            f"{feas.get('level', 'ambitious').replace('_', ' ')} versus past growth — "
            f"the longer timeline is the safer reading."
        )
    return (
        f"Plant {plant_id} is {fw_now:.1f} g on day {day_t}; your goal is {target_fw:.0f} g. "
        f"{tweak_bit}{caution} {time_bit} "
        f"Main reasons right now: {why}"
    )


def row_vector(bundle: ModelBundle, plant_id: int, day_t: int) -> tuple[pd.Series, np.ndarray]:
    df = bundle.pairs
    hit = df[(df["plant_id"] == plant_id) & (df["day_t"] == day_t)]
    if hit.empty:
        # allow day_future match as alias
        hit = df[(df["plant_id"] == plant_id) & (df["day_future"] == day_t + HORIZON)]
    if hit.empty:
        avail = df.loc[df["plant_id"] == plant_id, "day_t"].drop_duplicates().tolist()
        raise SystemExit(
            f"No pair for plant_id={plant_id} day_t={day_t}. Available day_t for plant: {avail}"
        )
    row = hit.iloc[0]
    x = row[bundle.cols].to_numpy(dtype=float).reshape(1, -1)
    return row, x


def predict(bundle: ModelBundle, x: np.ndarray) -> float:
    """AR-Δ: expected FW in HORIZON days = current fw + predicted change."""
    j = bundle.cols.index("fw_t")
    fw_now = float(x[0, j])
    delta = float(bundle.model.predict(x)[0])
    return max(0.0, fw_now + delta)


def predict_daily_roll(
    bundle: ModelBundle,
    plant_id: int,
    day_now: int,
    fw_now: float,
    n_days: int,
    climate_overrides: dict[str, float] | None = None,
) -> tuple[float, list[dict]]:
    """
    Professor-style autoregressive roll: t→t+1→… for n_days.
    Falls back to empty path if daily model missing.
    """
    if bundle.daily_model is None or bundle.daily_panel is None or not bundle.daily_feat_cols:
        return float("nan"), []
    climate_overrides = climate_overrides or {}
    # map recommender feature names → daily panel names where possible
    rename = {
        "env_day_air_temp_c_mean_wmean": "env_day_air_temp_c_mean",
        "env_night_air_temp_c_mean_wmean": "env_night_air_temp_c_mean",
        "env_day_vpd_pa_mean_wmean": "env_day_vpd_pa_mean",
        "env_day_co2_ppm_mean_wmean": "env_day_co2_ppm_mean",
        "nutrient_ec_us_cm_mean_wmean": "nutrient_ec_us_cm_mean",
    }
    panel = bundle.daily_panel
    fw = float(fw_now)
    gain = 0.0
    prev = panel[(panel["plant_id"] == plant_id) & (panel["DAT"] == day_now - 1)]
    if len(prev):
        gain = float(fw_now - float(prev.iloc[0].get("fw_interp", fw_now)))
    path = []
    for step in range(n_days):
        d = day_now + step
        row = panel[(panel["plant_id"] == plant_id) & (panel["DAT"] == d)]
        if row.empty:
            row = panel[panel["DAT"] == d].head(1)
        if row.empty:
            break
        r = row.iloc[0].copy()
        for feat, val in climate_overrides.items():
            key = rename.get(feat, feat)
            if key in r.index:
                r[key] = val
            if feat == "nutrient_ec_us_cm_mean_wmean" and "EC_uS_cm" in r.index:
                r["EC_uS_cm"] = val
            if feat == "env_day_air_temp_c_mean_wmean" and "T_air_light_C" in r.index:
                r["T_air_light_C"] = val
        r["fw_t"] = fw
        r["gain_prev_1d"] = gain
        r["DAT"] = float(d)
        x = r[bundle.daily_feat_cols].to_numpy(dtype=float).reshape(1, -1)
        dlt = float(np.clip(bundle.daily_model.predict(x)[0], -20.0, 80.0))
        fw_next = max(0.0, fw + dlt)
        path.append(
            {
                "from_day": d,
                "to_day": d + 1,
                "fw_start_g": round(fw, 2),
                "fw_pred_end_g": round(fw_next, 2),
                "gain_g": round(fw_next - fw, 2),
            }
        )
        gain = fw_next - fw
        fw = fw_next
    return fw, path


def clamp_action(bundle: ModelBundle, feat: str, value: float) -> tuple[float, bool]:
    """Clamp to [p05, p95] observed; return (value, was_clamped)."""
    r = bundle.ranges.get(feat)
    if not r:
        return value, False
    lo, hi = r["p05"], r["p95"]
    v = min(max(value, lo), hi)
    return v, v != value


def status(
    bundle: ModelBundle,
    plant_id: int,
    day_t: int,
    horizon: int = HORIZON,
) -> dict:
    row, x = row_vector(bundle, plant_id, day_t)
    fw_now = float(row["fw_t"])
    pred, path, mode = forecast_at_horizon(
        bundle, plant_id, int(row["day_t"]), horizon, x0=x, fw_now=fw_now
    )
    band = uncertainty_for_horizon(bundle.lodo_mae, horizon)
    # observed future only when horizon matches the table's +4d pair
    actual = None
    if horizon == HORIZON and np.isfinite(row.get("fw_future", np.nan)):
        actual = float(row["fw_future"])
    drivers = [d for d in bundle.importance if d["actionable"]][:5]
    context = [d for d in bundle.importance if not d["actionable"]][:3]
    shap_rows = local_shap(bundle, x, top_k=6)
    summary = (
        f"Plant {plant_id} weighs {fw_now:.1f} g today (day {int(row['day_t'])}). "
        f"In about {horizon} days (day {int(row['day_t']) + horizon}) it is expected "
        f"around {pred:.1f} g. Main reasons: {nlg_why_from_shap(shap_rows, top_k=3)}"
    )
    return {
        "query": "status",
        "plant_id": plant_id,
        "day_t": int(row["day_t"]),
        "day_future": int(row["day_t"]) + int(horizon),
        "horizon_days": int(horizon),
        "roll_mode": mode,
        "current_fw_g": fw_now,
        "predicted_fw_future_g": round(float(pred), 2),
        "observed_fw_future_g": None if actual is None else round(actual, 2),
        "path": path,
        "uncertainty_band_g": {
            "approx_mae_lodo": round(band, 2),
            "low": round(float(pred) - band, 2),
            "high": round(float(pred) + band, 2),
            "note": "± LODO MAE×√(horizon/4); not a formal CI",
        },
        "top_actionable_drivers": drivers,
        "strong_context_drivers": context,
        "shap_local": shap_rows,
        "shap_method": bundle.shap_method,
        "grower_summary": summary,
        "disclaimer": DISCLAIMER,
    }


def what_if(
    bundle: ModelBundle,
    plant_id: int,
    day_t: int,
    changes: dict[str, float],
    horizon: int = HORIZON,
) -> dict:
    row, x0 = row_vector(bundle, plant_id, day_t)
    fw_now = float(row["fw_t"])
    base, _, _ = forecast_at_horizon(
        bundle, plant_id, int(row["day_t"]), horizon, x0=x0, fw_now=fw_now
    )
    applied = []
    overrides = {}
    for feat, val in changes.items():
        if feat not in bundle.cols:
            raise SystemExit(f"Unknown feature {feat}. Actionable: {ACTIONABLE}")
        if feat not in ACTIONABLE:
            raise SystemExit(f"{feat} is not an actionable knob (context-only).")
        v, clamped = clamp_action(bundle, feat, float(val))
        j = bundle.cols.index(feat)
        old = float(x0[0, j])
        overrides[feat] = v
        applied.append(
            {
                "feature": feat,
                "name": FRIENDLY.get(feat, feat),
                "from": round(old, 3),
                "requested": float(val),
                "to": round(v, 3),
                "clamped_to_p05_p95": clamped,
            }
        )
    new, path, mode = forecast_at_horizon(
        bundle,
        plant_id,
        int(row["day_t"]),
        horizon,
        x0=x0,
        fw_now=fw_now,
        overrides=overrides,
    )
    x1 = apply_overrides(x0, bundle, overrides)
    shap_before = local_shap(bundle, x0, top_k=6)
    shap_after = local_shap(bundle, x1, top_k=6)
    summary = nlg_what_if(
        plant_id,
        int(row["day_t"]),
        base,
        new,
        uncertainty_for_horizon(bundle.lodo_mae, horizon),
        applied,
        shap_before,
        shap_after,
    )
    # fix horizon wording in summary if nlg hardcodes 4 — nlg_what_if uses HORIZON constant
    summary = summary.replace(f"in {HORIZON} days", f"in {horizon} days")
    return {
        "query": "what_if",
        "plant_id": plant_id,
        "day_t": int(row["day_t"]),
        "horizon_days": int(horizon),
        "roll_mode": mode,
        "baseline_predicted_fw_g": round(float(base), 2),
        "counterfactual_predicted_fw_g": round(float(new), 2),
        "delta_g": round(float(new) - float(base), 2),
        "changes": applied,
        "path": path,
        "shap_local_before": shap_before,
        "shap_local_after": shap_after,
        "shap_method": bundle.shap_method,
        "grower_summary": summary,
        "disclaimer": DISCLAIMER,
    }


def historical_gain_stats(bundle: ModelBundle) -> dict[str, float]:
    """Distribution of realized + predicted 4-day gains in the pair table."""
    g = (bundle.pairs["fw_future"] - bundle.pairs["fw_t"]).astype(float)
    g = g[np.isfinite(g)]
    return {
        "gain_p50_g": float(g.quantile(0.50)),
        "gain_p75_g": float(g.quantile(0.75)),
        "gain_p90_g": float(g.quantile(0.90)),
        "gain_p95_g": float(g.quantile(0.95)),
        "gain_max_g": float(g.max()),
    }


def apply_overrides(x: np.ndarray, bundle: ModelBundle, overrides: dict[str, float]) -> np.ndarray:
    xo = x.copy()
    for feat, val in overrides.items():
        if feat not in bundle.cols:
            continue
        v, _ = clamp_action(bundle, feat, float(val))
        xo[0, bundle.cols.index(feat)] = v
    return xo


def project_days_to_target(
    bundle: ModelBundle,
    x0: np.ndarray,
    fw_now: float,
    day_now: int,
    target_fw: float,
    overrides: dict[str, float] | None = None,
    max_day: int = 40,
    plant_id: int | None = None,
) -> dict:
    """
    Roll forward until predicted weight >= target.
    Prefer daily AR (t→t+1→…) when available; else coarse +4d AR-Δ steps.
    """
    overrides = overrides or {}
    fw = float(fw_now)
    day = int(day_now)
    path: list[dict] = []
    if fw >= target_fw:
        return {
            "already_at_or_above": True,
            "days_until_target": 0.0,
            "reach_on_approx_dat": float(day),
            "path": path,
            "reached": True,
            "roll_mode": "none",
        }

    # --- daily AR roll (professor path) ---
    if (
        plant_id is not None
        and bundle.daily_model is not None
        and bundle.daily_panel is not None
    ):
        fw_d = fw
        gain = 0.0
        panel = bundle.daily_panel
        prev = panel[(panel["plant_id"] == plant_id) & (panel["DAT"] == day - 1)]
        if len(prev):
            gain = float(fw_d - float(prev.iloc[0].get("fw_interp", fw_d)))
        rename = {
            "env_day_air_temp_c_mean_wmean": "env_day_air_temp_c_mean",
            "env_night_air_temp_c_mean_wmean": "env_night_air_temp_c_mean",
            "env_day_vpd_pa_mean_wmean": "env_day_vpd_pa_mean",
            "env_day_co2_ppm_mean_wmean": "env_day_co2_ppm_mean",
            "nutrient_ec_us_cm_mean_wmean": "nutrient_ec_us_cm_mean",
        }
        d = day
        while d <= max_day:
            row = panel[(panel["plant_id"] == plant_id) & (panel["DAT"] == d)]
            if row.empty:
                row = panel[panel["DAT"] == d].head(1)
            if row.empty:
                break
            r = row.iloc[0].copy()
            for feat, val in overrides.items():
                key = rename.get(feat, feat)
                if key in r.index:
                    r[key] = val
                if feat == "nutrient_ec_us_cm_mean_wmean" and "EC_uS_cm" in r.index:
                    r["EC_uS_cm"] = val
                if feat == "env_day_air_temp_c_mean_wmean" and "T_air_light_C" in r.index:
                    r["T_air_light_C"] = val
            r["fw_t"] = fw_d
            r["gain_prev_1d"] = gain
            r["DAT"] = float(d)
            x = r[bundle.daily_feat_cols].to_numpy(dtype=float).reshape(1, -1)
            dlt = float(np.clip(bundle.daily_model.predict(x)[0], -20.0, 80.0))
            fw_next = max(0.0, fw_d + dlt)
            path.append(
                {
                    "from_day": d,
                    "to_day": d + 1,
                    "fw_start_g": round(fw_d, 2),
                    "fw_pred_end_g": round(fw_next, 2),
                    "gain_g": round(fw_next - fw_d, 2),
                }
            )
            if fw_next >= target_fw:
                frac = 1.0 if dlt <= 1e-6 else float(np.clip((target_fw - fw_d) / dlt, 0.0, 1.0))
                days_needed = (d - day_now) + frac
                return {
                    "already_at_or_above": False,
                    "days_until_target": round(days_needed, 1),
                    "reach_on_approx_dat": round(day_now + days_needed, 1),
                    "path": path,
                    "reached": True,
                    "roll_mode": "daily_ar",
                }
            gain = fw_next - fw_d
            fw_d = fw_next
            d += 1
        return {
            "already_at_or_above": False,
            "days_until_target": None,
            "reach_on_approx_dat": None,
            "path": path,
            "reached": False,
            "roll_mode": "daily_ar",
            "note": f"Did not reach {target_fw:.0f} g by DAT≈{max_day} under daily AR.",
        }

    # --- fallback: +4d AR-Δ steps ---
    x = apply_overrides(x0, bundle, overrides)
    j_fw = bundle.cols.index("fw_t") if "fw_t" in bundle.cols else None
    j_gain = bundle.cols.index("gain_prev_4d") if "gain_prev_4d" in bundle.cols else None
    j_exp = bundle.cols.index("exp_day_wmean") if "exp_day_wmean" in bundle.cols else None

    while day <= max_day:
        if j_fw is not None:
            x[0, j_fw] = fw
        if j_exp is not None:
            x[0, j_exp] = float(day)
        pred = predict(bundle, x)
        gain = pred - fw
        path.append(
            {
                "from_day": day,
                "to_day": day + HORIZON,
                "fw_start_g": round(fw, 2),
                "fw_pred_end_g": round(pred, 2),
                "gain_g": round(gain, 2),
            }
        )
        if pred >= target_fw:
            if gain <= 1e-6:
                frac = 1.0
            else:
                frac = float(np.clip((target_fw - fw) / gain, 0.0, 1.0))
            days_needed = (day - day_now) + frac * HORIZON
            return {
                "already_at_or_above": False,
                "days_until_target": round(days_needed, 1),
                "reach_on_approx_dat": round(day_now + days_needed, 1),
                "path": path,
                "reached": True,
                "roll_mode": "ar_delta_4d",
            }
        if j_gain is not None:
            x[0, j_gain] = float(gain)
        fw = float(pred)
        day += HORIZON

    return {
        "already_at_or_above": False,
        "days_until_target": None,
        "reach_on_approx_dat": None,
        "path": path,
        "reached": False,
        "roll_mode": "ar_delta_4d",
        "note": f"Did not reach {target_fw:.0f} g by DAT≈{max_day} under this trajectory.",
    }


def feasibility_4d(
    fw_now: float,
    pred_4d: float,
    target_fw: float,
    gain_stats: dict[str, float],
    lodo_mae: float,
) -> dict:
    """Is hitting target in one +4d step historically / model-plausible?"""
    need_from_now = target_fw - fw_now
    need_from_pred = target_fw - pred_4d
    hist_p95 = gain_stats["gain_p95_g"]
    ambitious = need_from_now > hist_p95
    model_says_close = abs(need_from_pred) <= lodo_mae
    if need_from_now <= 0:
        level = "already_there"
        msg = "Plant is already at or above the target weight."
    elif need_from_now <= gain_stats["gain_p75_g"]:
        level = "plausible"
        msg = (
            f"Needing +{need_from_now:.0f} g in {HORIZON}d is within common past 4-day gains "
            f"(median +{gain_stats['gain_p50_g']:.0f} g, p75 +{gain_stats['gain_p75_g']:.0f} g)."
        )
    elif need_from_now <= hist_p95:
        level = "optimistic"
        msg = (
            f"Needing +{need_from_now:.0f} g in {HORIZON}d is high vs usual growth "
            f"(past p95 4-day gain ≈ +{hist_p95:.0f} g). Treat 4-day hit as optimistic."
        )
    else:
        level = "unlikely_in_4d"
        msg = (
            f"Needing +{need_from_now:.0f} g in {HORIZON}d exceeds almost all past 4-day gains "
            f"(p95 ≈ +{hist_p95:.0f} g, max ≈ +{gain_stats['gain_max_g']:.0f} g). "
            f"A model that still 'reaches' the target in 4d by cranking set-points is likely over-confident — "
            f"use the timeline (days-to-target) instead."
        )
    return {
        "level": level,
        "gain_needed_from_today_g": round(need_from_now, 2),
        "gap_after_model_4d_pred_g": round(need_from_pred, 2),
        "historical_4d_gain_p50_g": round(gain_stats["gain_p50_g"], 2),
        "historical_4d_gain_p95_g": round(hist_p95, 2),
        "model_4d_close_within_error": model_says_close,
        "message": msg,
    }


def when_target(
    bundle: ModelBundle,
    plant_id: int,
    day_t: int,
    target_fw: float,
) -> dict:
    """Only answer: under current settings, after how many days ≈ 100 g?"""
    row, x0 = row_vector(bundle, plant_id, day_t)
    fw_now = float(row["fw_t"])
    base = predict(bundle, x0)
    gains = historical_gain_stats(bundle)
    timeline = project_days_to_target(
        bundle, x0, fw_now, int(row["day_t"]), target_fw, plant_id=plant_id
    )
    feas = feasibility_4d(fw_now, base, target_fw, gains, bundle.lodo_mae)
    shap_rows = local_shap(bundle, x0, top_k=6)
    summary = nlg_when(
        plant_id,
        int(row["day_t"]),
        fw_now,
        base,
        target_fw,
        feas,
        timeline,
        shap_rows,
    )
    return {
        "query": "when_target",
        "plant_id": plant_id,
        "day_t": int(row["day_t"]),
        "target_fw_g": target_fw,
        "current_fw_g": round(fw_now, 2),
        "predicted_fw_in_4d_g": round(base, 2),
        "feasibility_for_4d": feas,
        "timeline_current_settings": timeline,
        "shap_local": shap_rows,
        "shap_method": bundle.shap_method,
        "grower_summary": summary,
        "disclaimer": DISCLAIMER,
    }


def reach_target(
    bundle: ModelBundle,
    plant_id: int,
    day_t: int,
    target_fw: float,
    max_knobs: int = 3,
    grid: int = 7,
) -> dict:
    """
    1) Suggest set-points to approach target in the next +4d forecast.
    2) Flag if that 4d goal is historically ambitious.
    3) Estimate days-to-target under current vs recommended settings.
    """
    row, x0 = row_vector(bundle, plant_id, day_t)
    fw_now = float(row["fw_t"])
    base = predict(bundle, x0)
    gains = historical_gain_stats(bundle)
    knobs = [d["feature"] for d in bundle.importance if d["actionable"]][:6]

    def eval_x(x):
        return abs(predict(bundle, x) - target_fw), predict(bundle, x)

    best = {
        "pred": base,
        "err": abs(base - target_fw),
        "changes": {},
        "x": x0.copy(),
    }

    for feat in knobs:
        r = bundle.ranges.get(feat)
        if not r:
            continue
        j = bundle.cols.index(feat)
        for v in np.linspace(r["p05"], r["p95"], grid):
            x = x0.copy()
            x[0, j] = float(v)
            err, pred = eval_x(x)
            if err < best["err"]:
                best = {"pred": pred, "err": err, "changes": {feat: float(v)}, "x": x}

    top = knobs[:4]
    if max_knobs >= 2:
        for i, f1 in enumerate(top):
            for f2 in top[i + 1 :]:
                r1, r2 = bundle.ranges.get(f1), bundle.ranges.get(f2)
                if not r1 or not r2:
                    continue
                j1, j2 = bundle.cols.index(f1), bundle.cols.index(f2)
                for v1 in np.linspace(r1["p05"], r1["p95"], 5):
                    for v2 in np.linspace(r2["p05"], r2["p95"], 5):
                        x = x0.copy()
                        x[0, j1] = float(v1)
                        x[0, j2] = float(v2)
                        err, pred = eval_x(x)
                        if err < best["err"]:
                            best = {
                                "pred": pred,
                                "err": err,
                                "changes": {f1: float(v1), f2: float(v2)},
                                "x": x,
                            }

    change_list = []
    for feat, val in best["changes"].items():
        j = bundle.cols.index(feat)
        change_list.append(
            {
                "feature": feat,
                "name": FRIENDLY.get(feat, feat),
                "from": round(float(x0[0, j]), 3),
                "to": round(float(val), 3),
                "observed_p50": round(bundle.ranges[feat]["p50"], 3),
            }
        )

    feas = feasibility_4d(fw_now, base, target_fw, gains, bundle.lodo_mae)
    # If 4d hit is unlikely, still show best 4d attempt but emphasize timeline
    timeline_cur = project_days_to_target(
        bundle,
        x0,
        fw_now,
        int(row["day_t"]),
        target_fw,
        overrides=None,
        plant_id=plant_id,
    )
    timeline_rec = project_days_to_target(
        bundle,
        x0,
        fw_now,
        int(row["day_t"]),
        target_fw,
        overrides=best["changes"],
        plant_id=plant_id,
    )

    reachable_4d = best["err"] <= bundle.lodo_mae and feas["level"] in (
        "already_there",
        "plausible",
        "optimistic",
    )

    shap_rows = local_shap(bundle, x0, top_k=6)
    summary = nlg_reach_target(
        plant_id,
        int(row["day_t"]),
        fw_now,
        target_fw,
        base,
        float(best["pred"]),
        feas,
        change_list,
        timeline_cur,
        timeline_rec,
        shap_rows,
        bundle.lodo_mae,
    )

    return {
        "query": "reach_target",
        "plant_id": plant_id,
        "day_t": int(row["day_t"]),
        "target_fw_g": target_fw,
        "current_fw_g": round(fw_now, 2),
        "baseline_predicted_fw_g": round(base, 2),
        "recommended_predicted_fw_g": round(best["pred"], 2),
        "gap_to_target_g": round(best["pred"] - target_fw, 2),
        "recommendations": change_list,
        "feasibility_for_4d": feas,
        "within_typical_error": bool(reachable_4d),
        "timeline_current_settings": timeline_cur,
        "timeline_recommended_settings": timeline_rec,
        "shap_local": shap_rows,
        "shap_method": bundle.shap_method,
        "grower_summary": summary,
        "disclaimer": DISCLAIMER,
    }


def report_day(bundle: ModelBundle, day_t: int, target_fw: float | None) -> pd.DataFrame:
    df = bundle.pairs
    plants = sorted(df.loc[df["day_t"] == day_t, "plant_id"].unique())
    if not plants:
        raise SystemExit(f"No plants with day_t={day_t}")
    rows = []
    for pid in plants:
        st = status(bundle, int(pid), day_t)
        rec = {
            "plant_id": pid,
            "day_t": day_t,
            "fw_t": st["current_fw_g"],
            "pred_fw_future": st["predicted_fw_future_g"],
            "obs_fw_future": st["observed_fw_future_g"],
            "band_low": st["uncertainty_band_g"]["low"],
            "band_high": st["uncertainty_band_g"]["high"],
            "top_driver_1": st["top_actionable_drivers"][0]["name"]
            if st["top_actionable_drivers"]
            else "",
            "summary": st["grower_summary"],
        }
        if target_fw is not None:
            rt = reach_target(bundle, int(pid), day_t, target_fw)
            rec["target_fw"] = target_fw
            rec["recommended_pred"] = rt["recommended_predicted_fw_g"]
            rec["n_changes"] = len(rt["recommendations"])
            rec["rec_text"] = rt["grower_summary"]
        rows.append(rec)
    return pd.DataFrame(rows)


def parse_sets(items: list[str]) -> dict[str, float]:
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"Bad --set '{it}', expected feature=value")
        k, v = it.split("=", 1)
        out[k.strip()] = float(v)
    return out


def resolve_horizon(day_t: int, horizon: int | None, until_day: int | None) -> int:
    """User may pass days-ahead (--horizon) or absolute DAT (--until-day)."""
    if until_day is not None:
        h = int(until_day) - int(day_t)
        if h <= 0:
            raise SystemExit(
                f"--until-day ({until_day}) must be after --day ({day_t})."
            )
        return h
    if horizon is None:
        return HORIZON
    h = int(horizon)
    if h <= 0:
        raise SystemExit("--horizon must be a positive number of days.")
    if h > 40:
        raise SystemExit("--horizon too large (max 40 days for this trial window).")
    return h


def forecast_at_horizon(
    bundle: ModelBundle,
    plant_id: int,
    day_t: int,
    horizon: int,
    x0: np.ndarray | None = None,
    fw_now: float | None = None,
    overrides: dict[str, float] | None = None,
) -> tuple[float, list[dict], str]:
    """
    Predict FW after `horizon` days.
      - horizon == 4: AR-Δ (best LODO accuracy)
      - other horizons: daily AR roll t→t+1→… when available
      - multiples of 4 without daily model: repeated AR-Δ steps
    """
    if x0 is None or fw_now is None:
        row, x0 = row_vector(bundle, plant_id, day_t)
        fw_now = float(row["fw_t"])
    x = apply_overrides(x0, bundle, overrides or {})
    j = bundle.cols.index("fw_t")
    # keep fw_now authoritative if provided
    x[0, j] = float(fw_now)

    if horizon == HORIZON and not overrides:
        pred = predict(bundle, x)
        path = [
            {
                "from_day": day_t,
                "to_day": day_t + horizon,
                "fw_start_g": round(fw_now, 2),
                "fw_pred_end_g": round(pred, 2),
                "gain_g": round(pred - fw_now, 2),
            }
        ]
        return pred, path, "ar_delta_4d"

    if horizon == HORIZON and overrides:
        pred = predict(bundle, x)
        path = [
            {
                "from_day": day_t,
                "to_day": day_t + horizon,
                "fw_start_g": round(fw_now, 2),
                "fw_pred_end_g": round(pred, 2),
                "gain_g": round(pred - fw_now, 2),
            }
        ]
        return pred, path, "ar_delta_4d"

    if bundle.daily_model is not None and horizon > 0:
        pred, path = predict_daily_roll(
            bundle, plant_id, day_t, fw_now, horizon, climate_overrides=overrides
        )
        if np.isfinite(pred) and path:
            return float(pred), path, "daily_ar"

    if horizon % HORIZON == 0:
        steps = horizon // HORIZON
        fw = float(fw_now)
        xx = x.copy()
        path = []
        j_fw = bundle.cols.index("fw_t")
        j_gain = (
            bundle.cols.index("gain_prev_4d") if "gain_prev_4d" in bundle.cols else None
        )
        j_exp = (
            bundle.cols.index("exp_day_wmean") if "exp_day_wmean" in bundle.cols else None
        )
        day = int(day_t)
        for _ in range(steps):
            if j_fw is not None:
                xx[0, j_fw] = fw
            if j_exp is not None:
                xx[0, j_exp] = float(day)
            pred_s = predict(bundle, xx)
            path.append(
                {
                    "from_day": day,
                    "to_day": day + HORIZON,
                    "fw_start_g": round(fw, 2),
                    "fw_pred_end_g": round(pred_s, 2),
                    "gain_g": round(pred_s - fw, 2),
                }
            )
            if j_gain is not None:
                xx[0, j_gain] = pred_s - fw
            fw = pred_s
            day += HORIZON
        return float(fw), path, "ar_delta_4d_multistep"

    raise SystemExit(
        f"Cannot forecast horizon={horizon}: daily AR model missing. "
        f"Use --horizon {HORIZON} or a multiple of {HORIZON}, or train daily AR artifacts."
    )


def uncertainty_for_horizon(lodo_mae_4d: float, horizon: int) -> float:
    """Rough error band: scale 4d LODO MAE with sqrt(horizon/4)."""
    return float(lodo_mae_4d) * float(np.sqrt(max(horizon, 1) / HORIZON))


def predict_future(
    bundle: ModelBundle,
    plant_id: int,
    day_t: int,
    horizon: int,
    fw_today: float | None = None,
) -> dict:
    """
    Predict weight after `horizon` days when today's weight is known
    (or taken from the table) and future weight is unknown.
    """
    row, x0 = row_vector(bundle, plant_id, day_t)
    if fw_today is not None:
        j = bundle.cols.index("fw_t")
        x0 = x0.copy()
        x0[0, j] = float(fw_today)
        fw_now = float(fw_today)
    else:
        fw_now = float(row["fw_t"])

    pred, path, mode = forecast_at_horizon(
        bundle, plant_id, int(row["day_t"]), horizon, x0=x0, fw_now=fw_now
    )
    band = uncertainty_for_horizon(bundle.lodo_mae, horizon)
    shap_rows = local_shap(bundle, x0, top_k=5)
    target_day = int(row["day_t"]) + int(horizon)
    summary = (
        f"Plant {plant_id} is {fw_now:.1f} g on day {int(row['day_t'])}. "
        f"After {horizon} days (around day {target_day}) it is expected around {pred:.1f} g "
        f"(rough range {pred - band:.0f}–{pred + band:.0f} g). "
        f"Main reasons: {nlg_why_from_shap(shap_rows, top_k=2)}"
    )
    return {
        "query": "predict",
        "plant_id": plant_id,
        "day_t": int(row["day_t"]),
        "target_day": target_day,
        "horizon_days": int(horizon),
        "current_fw_g": round(fw_now, 2),
        "predicted_fw_g": round(float(pred), 2),
        "uncertainty_band_g": {
            "approx_mae_scaled": round(band, 2),
            "low": round(float(pred) - band, 2),
            "high": round(float(pred) + band, 2),
            "note": "± LODO MAE×√(horizon/4); not a formal CI",
        },
        "roll_mode": mode,
        "path": path,
        "shap_local": shap_rows,
        "grower_summary": summary,
        "disclaimer": DISCLAIMER,
    }


def _hr(title: str) -> str:
    return f"\n{'═' * 60}\n  {title}\n{'═' * 60}"


def _format_shap_lines(shap_rows: list[dict] | None, method: str | None = None, indent: str = "  ") -> list[str]:
    lines = []
    if not shap_rows:
        return lines
    lines.append(f"{indent}What’s helping or slowing this plant:")
    for r in shap_rows[:5]:
        arrow = "↑ helping" if r.get("pushes") == "up" else "↓ slowing"
        knob = "  (adjustable)" if r.get("actionable") else ""
        lines.append(
            f"{indent}  {arrow}: {r['name']} (~{abs(r['shap_g']):.1f} g){knob}"
        )
    return lines


def format_human(out: dict, bundle: ModelBundle | None = None) -> str:
    """Plain-English report for terminal (in addition to JSON)."""
    q = out.get("query", "")
    lines: list[str] = []

    if q == "status":
        lines.append(_hr("PLANT STATUS — next weigh-in forecast"))
        lines.append(f"  Plant ID          : {out['plant_id']}")
        lines.append(f"  Today (DAT)       : day {out['day_t']}")
        lines.append(
            f"  Forecast day       : day {out['day_future']}  (+{out['horizon_days']} days)"
        )
        lines.append("")
        lines.append(f"  Weight today      : {out['current_fw_g']:.1f} g")
        lines.append(f"  Expected then     : {out['predicted_fw_future_g']:.1f} g")
        if out.get("observed_fw_future_g") is not None:
            err = abs(out["predicted_fw_future_g"] - out["observed_fw_future_g"])
            lines.append(
                f"  Actually weighed  : {out['observed_fw_future_g']:.1f} g  "
                f"(off by {err:.1f} g)"
            )
        band = out["uncertainty_band_g"]
        lines.append("")
        lines.append(
            f"  Likely range in {out['horizon_days']} days : "
            f"{band['low']:.0f} – {band['high']:.0f} g"
        )
        if out.get("path") and out.get("horizon_days", HORIZON) != HORIZON:
            lines.append("  Day-by-day path:")
            for step in (out.get("path") or [])[:8]:
                lines.append(
                    f"    day {step['from_day']}→{step['to_day']}: "
                    f"{step['fw_start_g']:.1f} → {step['fw_pred_end_g']:.1f} g"
                )
        lines.append("")
        lines.extend(_format_shap_lines(out.get("shap_local")))
        lines.append("")
        lines.append("  Settings worth watching:")
        for i, d in enumerate(out.get("top_actionable_drivers") or [], 1):
            direction = "higher → heavier" if d.get("coef", 0) > 0 else "higher → lighter"
            lines.append(f"    {i}. {d['name']}  ({direction})")
        lines.append("")
        lines.append(f"  Summary: {out.get('grower_summary', '')}")

    elif q == "what_if":
        lines.append(_hr("WHAT-IF — pretend you change tent settings"))
        lines.append(f"  Plant ID          : {out['plant_id']}   (day {out['day_t']})")
        lines.append(f"  Expected BEFORE   : {out['baseline_predicted_fw_g']:.1f} g in {HORIZON} days")
        lines.append(f"  Expected AFTER    : {out['counterfactual_predicted_fw_g']:.1f} g")
        delta = out["delta_g"]
        arrow = "UP" if delta > 0 else ("DOWN" if delta < 0 else "SAME")
        lines.append(f"  Change            : {delta:+.1f} g  ({arrow})")
        lines.append("")
        lines.append("  Settings you asked for:")
        for c in out.get("changes") or []:
            clamp_note = "  ⚠ pulled back into normal past range (p05–p95)" if c.get("clamped_to_p05_p95") else ""
            lines.append(
                f"    • {c['name']}: {c['from']:.2f} → {c['to']:.2f}"
                f"  (you typed {c['requested']}){clamp_note}"
            )
        lines.append("")
        lines.extend(_format_shap_lines(out.get("shap_local_after")))
        lines.append("")
        lines.append(f"  Summary: {out.get('grower_summary', '')}")

    elif q == "when_target":
        lines.append(_hr("WHEN WILL IT REACH THE TARGET WEIGHT?"))
        lines.append(f"  Plant ID             : {out['plant_id']}   (day {out['day_t']})")
        lines.append(f"  Weight today         : {out['current_fw_g']:.1f} g")
        lines.append(f"  Your goal            : {out['target_fw_g']:.0f} g")
        lines.append(f"  Expected in +{HORIZON}d only   : {out['predicted_fw_in_4d_g']:.1f} g")
        feas = out.get("feasibility_for_4d") or {}
        feas_msg = (feas.get("message") or "").replace(
            "A model that still 'reaches' the target in 4d by cranking set-points is likely over-confident — "
            "use the timeline (days-to-target) instead.",
            "Prefer the longer timeline over assuming a single 4-day jump.",
        )
        lines.append("")
        lines.append(f"  Is +{HORIZON}d enough vs past growth?")
        lines.append(f"    Verdict : {feas.get('level', '?')}")
        lines.append(f"    Detail  : {feas_msg}")
        tl = out.get("timeline_current_settings") or {}
        lines.append("")
        lines.append("  Estimated timeline under current settings:")
        if tl.get("days_until_target") is not None:
            lines.append(
                f"    ≈ {tl['days_until_target']:.1f} days from today"
                f"  (around day {tl['reach_on_approx_dat']})"
            )
        else:
            lines.append(f"    {tl.get('note', 'Not reached in the projection window.')}")
        if tl.get("path"):
            mode = tl.get("roll_mode", "")
            mode_note = " (day-by-day)" if mode == "daily_ar" else ""
            lines.append(f"    Step-by-step growth path{mode_note}:")
            steps = tl["path"]
            if len(steps) <= 8:
                show = steps
            else:
                show = steps[:4] + [None] + steps[-3:]
            for step in show:
                if step is None:
                    lines.append("      …")
                    continue
                lines.append(
                    f"      day {step['from_day']}→{step['to_day']}: "
                    f"{step['fw_start_g']:.1f} g → {step['fw_pred_end_g']:.1f} g "
                    f"(+{step['gain_g']:.1f} g)"
                )
        lines.append("")
        lines.extend(_format_shap_lines(out.get("shap_local")))
        lines.append("")
        lines.append(f"  Summary: {out.get('grower_summary', '')}")

    elif q == "reach_target":
        lines.append(_hr("GOAL WEIGHT — changes for next weigh-in + when you'll get there"))
        lines.append(f"  Plant ID             : {out['plant_id']}   (day {out['day_t']})")
        lines.append(f"  Weight today         : {out.get('current_fw_g', float('nan')):.1f} g")
        lines.append(f"  Your goal            : {out['target_fw_g']:.0f} g")
        lines.append("")
        lines.append(f"  --- A) Next {HORIZON} days only ---")
        lines.append(f"  Expected in +{HORIZON}d (no change) : {out['baseline_predicted_fw_g']:.1f} g")
        lines.append(f"  Expected in +{HORIZON}d (w/ tweaks): {out['recommended_predicted_fw_g']:.1f} g")
        feas = out.get("feasibility_for_4d") or {}
        lines.append(f"  Realism of hitting goal in {HORIZON}d: {feas.get('level', '?')}")
        feas_msg = (feas.get("message") or "").replace(
            "A model that still 'reaches' the target in 4d by cranking set-points is likely over-confident — "
            "use the timeline (days-to-target) instead.",
            "Prefer the longer timeline over assuming a single 4-day jump.",
        )
        lines.append(f"    {feas_msg}")
        lines.append("")
        recs = out.get("recommendations") or []
        if not recs:
            lines.append("  Suggested setting changes: none found inside safe past ranges.")
        else:
            warn = (
                "  (Use cautiously — 4-day goal looks historically unlikely.)"
                if feas.get("level") == "unlikely_in_4d"
                else ""
            )
            lines.append(f"  Suggested setting changes{warn}:")
            for c in recs:
                lines.append(
                    f"    • {c['name']}: {c['from']:.2f} → {c['to']:.2f}"
                    f"   (past median {c['observed_p50']:.2f})"
                )
        lines.append("")
        lines.append("  --- B) After how many days ≈ goal? ---")
        tl_c = out.get("timeline_current_settings") or {}
        tl_r = out.get("timeline_recommended_settings") or {}
        if tl_c.get("days_until_target") is not None:
            lines.append(
                f"  Current settings : ≈ {tl_c['days_until_target']:.1f} days "
                f"(around day {tl_c['reach_on_approx_dat']})"
            )
        else:
            lines.append(f"  Current settings : {tl_c.get('note', 'not reached in window')}")
        if tl_r.get("days_until_target") is not None and recs:
            lines.append(
                f"  With suggestions : ≈ {tl_r['days_until_target']:.1f} days "
                f"(around day {tl_r['reach_on_approx_dat']})"
            )
        elif recs:
            lines.append(f"  With suggestions : {tl_r.get('note', 'not reached in window')}")
        if tl_c.get("path"):
            mode = tl_c.get("roll_mode", "")
            mode_note = " (day-by-day)" if mode == "daily_ar" else ""
            lines.append(f"  Growth path (current settings){mode_note}:")
            steps = tl_c["path"]
            # show up to 8 steps; if daily and longer, show head+tail
            if len(steps) <= 8:
                show = steps
            else:
                show = steps[:4] + [{"from_day": "…", "to_day": "…", "fw_start_g": float("nan"), "fw_pred_end_g": float("nan")}] + steps[-3:]
            for step in show:
                if step["from_day"] == "…":
                    lines.append("    …")
                    continue
                lines.append(
                    f"    day {step['from_day']}→{step['to_day']}: "
                    f"{step['fw_start_g']:.1f} → {step['fw_pred_end_g']:.1f} g"
                )
        lines.append("")
        lines.extend(_format_shap_lines(out.get("shap_local")))
        lines.append("")
        lines.append(f"  Summary: {out.get('grower_summary', '')}")

    elif q == "predict":
        lines.append(_hr("PREDICT FUTURE WEIGHT (today known, future unknown)"))
        lines.append(f"  Plant ID             : {out['plant_id']}   (day {out['day_t']})")
        lines.append(f"  Weight today         : {out['current_fw_g']:.1f} g")
        lines.append(f"  Horizon              : +{out['horizon_days']} days")
        if out.get("target_day") is not None:
            lines.append(f"  Target day (DAT)     : {out['target_day']}")
        lines.append(f"  Expected then        : {out['predicted_fw_g']:.1f} g")
        band = out.get("uncertainty_band_g") or {}
        if band:
            lines.append(
                f"  Likely range         : {band.get('low', float('nan')):.0f} – "
                f"{band.get('high', float('nan')):.0f} g"
            )
        lines.append("")
        if out.get("path"):
            lines.append("  Growth path:")
            steps = out["path"]
            show = steps if len(steps) <= 10 else steps[:4] + [None] + steps[-3:]
            for step in show:
                if step is None:
                    lines.append("    …")
                    continue
                lines.append(
                    f"    day {step['from_day']}→{step['to_day']}: "
                    f"{step['fw_start_g']:.1f} → {step['fw_pred_end_g']:.1f} g"
                )
        lines.append("")
        lines.extend(_format_shap_lines(out.get("shap_local")))
        lines.append("")
        lines.append(f"  Summary: {out.get('grower_summary', '')}")

    else:
        lines.append(json.dumps(out, indent=2))

    lines.append("")
    lines.append("─" * 60)
    return "\n".join(lines)


def format_report_human(df: pd.DataFrame, day_t: int, target_fw: float | None, lodo_mae: float) -> str:
    lines = [_hr(f"TRAY REPORT — all plants on day {day_t}")]
    lines.append(f"  Plants on this day : {len(df)}")
    lines.append(f"  Typical forecast error (LODO MAE): ±{lodo_mae:.0f} g")
    if target_fw is not None:
        lines.append(f"  Goal weight you set: {target_fw:.0f} g")
    lines.append("")
    lines.append(
        f"  {'Plant':>6}  {'Today g':>8}  {'Pred +4d':>9}  {'Actual':>8}  {'Miss':>7}  {'Rough range':>16}"
    )
    lines.append("  " + "-" * 64)
    for _, r in df.iterrows():
        actual = r.get("obs_fw_future")
        if actual is not None and pd.notna(actual):
            miss = abs(float(r["pred_fw_future"]) - float(actual))
            actual_s = f"{float(actual):8.1f}"
            miss_s = f"{miss:7.1f}"
        else:
            actual_s = f"{'—':>8}"
            miss_s = f"{'—':>7}"
        lines.append(
            f"  {int(r['plant_id']):6d}  {float(r['fw_t']):8.1f}  {float(r['pred_fw_future']):9.1f}  "
            f"{actual_s}  {miss_s}  {float(r['band_low']):5.0f}–{float(r['band_high']):<5.0f} g"
        )
    lines.append("")
    lines.append("  Columns:")
    lines.append("    Today g   = weight now")
    lines.append("    Pred +4d  = model forecast for next weigh-in")
    lines.append("    Actual    = real weight if already measured (to check the model)")
    lines.append("    Miss      = |Pred − Actual| for that plant")
    lines.append("    Range     = Pred ± typical LODO error")
    if target_fw is not None and "rec_text" in df.columns:
        lines.append("")
        lines.append("  Per-plant suggestion toward your goal (short):")
        for _, r in df.iterrows():
            lines.append(f"    Plant {int(r['plant_id'])}: predicted→{r.get('recommended_pred', '—')} g after tweaks")
    lines.append("")
    lines.append("─" * 60)
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Greenhouse growth recommendation CLI")
    p.add_argument(
        "--json",
        action="store_true",
        help="Also print raw JSON after the human-readable summary",
    )
    p.add_argument(
        "--trial",
        choices=["trial1", "trial2"],
        default="trial1",
        help="trial1=per-plant pairs; trial2=tray-median pairs (plant-id 0)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--plant-id", type=int, required=True)
        sp.add_argument("--day", type=int, required=True, help="day_t (DAT of current state)")
        sp.add_argument("--seed", type=int, default=42)

    def add_horizon(sp):
        sp.add_argument(
            "--horizon",
            type=int,
            default=None,
            help="days ahead to forecast (e.g. 3, 4, 5, 7). Default 4 if omitted.",
        )
        sp.add_argument(
            "--until-day",
            type=int,
            default=None,
            help="absolute target DAT (e.g. 17). Alternative to --horizon.",
        )

    sp = sub.add_parser("status", help="Forecast + top drivers for one plant")
    add_common(sp)
    add_horizon(sp)

    sp = sub.add_parser("what_if", help="Counterfactual set-point changes")
    add_common(sp)
    add_horizon(sp)
    sp.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        help="feature=value (repeatable). Features: see ACTIONABLE list.",
    )

    sp = sub.add_parser(
        "reach_target",
        help="Goal weight: suggested changes for +4d AND estimated days to reach goal",
    )
    add_common(sp)
    sp.add_argument("--target-fw", type=float, required=True)

    sp = sub.add_parser(
        "when",
        help="Only ask: after how many days will the plant reach --target-fw?",
    )
    add_common(sp)
    sp.add_argument("--target-fw", type=float, required=True)

    sp = sub.add_parser("report", help="Batch CSV for all plants on a day")
    sp.add_argument("--day", type=int, required=True)
    sp.add_argument("--target-fw", type=float, default=None)
    sp.add_argument("--seed", type=int, default=42)

    sp = sub.add_parser(
        "predict",
        help="Predict future weight for a user-chosen horizon / target day",
    )
    add_common(sp)
    add_horizon(sp)
    sp.add_argument(
        "--fw-today",
        type=float,
        default=None,
        help="optional override for today's weight (g); else use table value",
    )

    sp = sub.add_parser("list-days", help="Show available plant/day pairs")
    sp.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = build_bundle(seed=args.seed, trial=args.trial)

    if args.cmd == "list-days":
        g = (
            bundle.pairs.groupby(["plant_id", "day_t"], as_index=False)
            .size()
            .sort_values(["day_t", "plant_id"])
        )
        print(_hr("AVAILABLE PLANT / DAY PAIRS"))
        print("  Use --plant-id and --day from this list.\n")
        print(g.to_string(index=False))
        print(
            f"\n  Model typical error (LODO MAE): ±{bundle.lodo_mae:.1f} g on future-weight forecasts."
        )
        return

    if args.cmd == "status":
        h = resolve_horizon(args.day, args.horizon, args.until_day)
        out = status(bundle, args.plant_id, args.day, horizon=h)
    elif args.cmd == "what_if":
        if not args.sets:
            raise SystemExit("Pass at least one --set feature=value")
        h = resolve_horizon(args.day, args.horizon, args.until_day)
        out = what_if(
            bundle, args.plant_id, args.day, parse_sets(args.sets), horizon=h
        )
    elif args.cmd == "reach_target":
        out = reach_target(bundle, args.plant_id, args.day, args.target_fw)
    elif args.cmd == "when":
        out = when_target(bundle, args.plant_id, args.day, args.target_fw)
    elif args.cmd == "predict":
        h = resolve_horizon(args.day, args.horizon, args.until_day)
        out = predict_future(
            bundle,
            args.plant_id,
            args.day,
            h,
            fw_today=args.fw_today,
        )
    elif args.cmd == "report":
        df = report_day(bundle, args.day, args.target_fw)
        path = OUT / f"recommendations_day{args.day:02d}.csv"
        df.to_csv(path, index=False)
        meta = {
            "day_t": args.day,
            "n_plants": int(len(df)),
            "lodo_mae_g": bundle.lodo_mae,
            "target_fw": args.target_fw,
            "disclaimer": DISCLAIMER,
            "csv": str(path),
        }
        (OUT / f"recommendations_day{args.day:02d}_meta.json").write_text(
            json.dumps(meta, indent=2)
        )
        print(format_report_human(df, args.day, args.target_fw, bundle.lodo_mae))
        print(f"\n  Full table also saved to:\n    {path}")
        if args.json:
            print("\n" + df.to_json(orient="records", indent=2))
        return
    else:
        raise SystemExit("unknown command")

    print(format_human(out, bundle))
    (OUT / f"last_{args.cmd}.json").write_text(json.dumps(out, indent=2))
    (OUT / f"last_{args.cmd}.txt").write_text(format_human(out, bundle))
    if args.json:
        print("\n--- raw JSON ---")
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

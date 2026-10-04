#!/usr/bin/env python3
"""
Paper-grade factor attribution for Aim 1 / future-FW models.

Primary methods (always available):
  - permutation importance (row shuffle) + day-block permutation
  - 1D partial dependence (PDP)

SHAP (when importable; needs shap built for current NumPy):
  - LinearExplainer for ElasticNet forecast
  - TreeExplainer / Explainer for XGB/GBM nowcast
  Rankings are compared to |coef| / permutation; SHAP does not change LODO MAE.

Targets:
  - nowcast: compact features @ t → FW @ t  (Section 10/12 style)
  - forecast: sensors + fw_t @ t → FW @ t+4 (Aim 2 direct route)

Usage:
  python3 ml/fw_attribution.py --task both --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import partial_dependence, permutation_importance
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
PLANT = ML / "plant"
OUT = ML / "fw_attribution"
HORIZON = 4

NOWCAST_LIN_PHY = [
    "day",
    "plant_id",
    "plant_img_mask_area_px_last",
    "plant_img_mask_area_px_wmean",
    "plant_img_mask_area_px_delta",
    "plant_img_plant_color_frac_last",
    "plant_img_plant_color_frac_wmean",
    "img_mask_area_mean_wmean",
    "img_mask_area_sum_wmean",
    "img_coverage_frac_wmean",
    "img_plant_color_mean_wmean",
    "win_img_mask_area_sum_slope",
    "win_img_mask_area_sum_delta",
    "win_img_coverage_frac_slope",
    "env_day_air_temp_c_mean_wmean",
    "env_night_air_temp_c_mean_wmean",
    "env_day_humidity_pct_mean_wmean",
    "env_night_humidity_pct_mean_wmean",
    "env_day_co2_ppm_mean_wmean",
    "env_night_co2_ppm_mean_wmean",
    "env_day_vpd_pa_mean_wmean",
    "env_night_vpd_pa_mean_wmean",
    "nutrient_ec_us_cm_mean_wmean",
    "nutrient_ph_mean_wmean",
    "env_act_day_light_duration_s_sum_wmean",
    "day_sq",
    "log_mask_area",
]

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

# PDP focus features (actionable / climate)
PDP_NOWCAST = [
    "nutrient_ec_us_cm_mean_wmean",
    "env_day_co2_ppm_mean_wmean",
    "env_day_vpd_pa_mean_wmean",
    "env_act_day_light_duration_s_sum_wmean",
    "plant_img_mask_area_px_last",
]
PDP_FORECAST = [
    "nutrient_ec_us_cm_mean_wmean",
    "env_day_co2_ppm_mean_wmean",
    "env_day_vpd_pa_mean_wmean",
    "env_act_day_light_duration_s_sum_wmean",
    "fw_t",
]


def xgb_or_gbm(seed: int):
    try:
        from xgboost import XGBRegressor

        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=120,
                        max_depth=3,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        reg_lambda=2.0,
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    except Exception:
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=300, max_depth=2, random_state=seed
                    ),
                ),
            ]
        )


def elastic(seed: int):
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


def load_nowcast() -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    df = pd.read_csv(PLANT / "plant_checkpoint_trial1_compact.csv").copy()
    area = pd.to_numeric(df.get("plant_img_mask_area_px_last"), errors="coerce")
    day = pd.to_numeric(df["day"], errors="coerce")
    df["day_sq"] = day**2
    df["log_mask_area"] = np.log1p(area.clip(lower=0))
    cols = [c for c in NOWCAST_LIN_PHY if c in df.columns]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    y = pd.to_numeric(df["target_fw_g"], errors="coerce").to_numpy(dtype=float)
    days = day.to_numpy(dtype=int)
    return df, cols, y, days


def load_forecast_pairs() -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    # Prefer Aim2 pairs if present; else build lightly from compact
    for cand in (
        ML / "future_fw_paper" / "plant_pairs_aim2.csv",
        ML / "aim2_phenotype_tuned" / "plant_pairs_aim2.csv",
    ):
        if cand.exists():
            pairs = pd.read_csv(cand)
            break
    else:
        # minimal rebuild
        import runpy

        api = runpy.run_path(str(ML / "train_aim2_phenotype_consistency.py"))
        pairs = api["build_pairs"]()

    cols = [c for c in FORECAST_COLS if c in pairs.columns]
    for c in cols:
        pairs[c] = pd.to_numeric(pairs[c], errors="coerce")
    pairs = pairs[np.isfinite(pairs["fw_future"].to_numpy(dtype=float))].copy()
    y = pairs["fw_future"].to_numpy(dtype=float)
    days = pairs["day_future"].to_numpy(dtype=int)
    return pairs, cols, y, days


def perm_importance_full(
    model, X: np.ndarray, y: np.ndarray, feature_names: list[str], seed: int
) -> pd.DataFrame:
    """
    Fit on all rows, then permutation importance on the same set.
    Paper caveat: explanatory ranking (not LODO predictive skill). Tray-level
    climate features are constant within a harvest day, so LODO-held-out
    permutation cannot credit them — hence full-fit attribution.
    """
    m = clone(model)
    m.fit(X, y)
    r = permutation_importance(
        m,
        X,
        y,
        n_repeats=30,
        random_state=seed,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    rows = []
    for i, c in enumerate(feature_names):
        # score = -MAE; positive importance ≈ MAE increase when shuffled
        rows.append(
            {
                "feature": c,
                "mae_increase_mean": float(r.importances_mean[i]),
                "mae_increase_sd": float(r.importances_std[i]),
                "n_repeats": 30,
            }
        )
    return pd.DataFrame(rows).sort_values("mae_increase_mean", ascending=False)


def day_block_perm_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    days: np.ndarray,
    feature_names: list[str],
    seed: int,
    n_repeats: int = 40,
) -> pd.DataFrame:
    """
    Shuffle whole harvest-day blocks for one feature at a time.
    Credits tray-level climate/actuator drivers that LODO within-day shuffle cannot.
    """
    rng = np.random.default_rng(seed)
    m = clone(model)
    m.fit(X, y)
    base = float(np.mean(np.abs(y - m.predict(X))))
    uniq = np.unique(days)
    rows = []
    for j, c in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            # permute feature values by day: assign each day's vector to a random day
            day_vals = {d: X[days == d, j].copy() for d in uniq}
            # for tray-level feats all rows in a day equal; still permute day→day mapping
            perm_days = rng.permutation(uniq)
            mapping = {d: perm_days[i] for i, d in enumerate(uniq)}
            for d in uniq:
                src = mapping[d]
                # broadcast source day's mean (tray) or per-plant values if same length
                src_vals = day_vals[src]
                n_d = int((days == d).sum())
                if len(src_vals) == n_d:
                    Xp[days == d, j] = src_vals
                else:
                    Xp[days == d, j] = float(np.mean(src_vals))
            mae = float(np.mean(np.abs(y - m.predict(Xp))))
            deltas.append(mae - base)
        rows.append(
            {
                "feature": c,
                "mae_increase_mean": float(np.mean(deltas)),
                "mae_increase_sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "n_repeats": n_repeats,
                "method": "day_block",
            }
        )
    return pd.DataFrame(rows).sort_values("mae_increase_mean", ascending=False)

def fit_full_and_pdp(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    pdp_feats: list[str],
    out_png: Path,
    title: str,
):
    m = clone(model)
    m.fit(X, y)
    feats = [f for f in pdp_feats if f in feature_names]
    if not feats:
        return
    idxs = [feature_names.index(f) for f in feats]
    n = len(feats)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), squeeze=False)
    for ax, f, idx in zip(axes[0], feats, idxs):
        pd_res = partial_dependence(m, X, [idx], kind="average", grid_resolution=40)
        ax.plot(pd_res["grid_values"][0], pd_res["average"][0], lw=2, color="#1b5e20")
        ax.set_xlabel(f.replace("_wmean", "").replace("env_", "").replace("nutrient_", "")[:40])
        ax.set_ylabel("partial dep. (g)")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def nice_label(c: str) -> str:
    return (
        c.replace("_wmean", "")
        .replace("env_day_", "day ")
        .replace("env_night_", "night ")
        .replace("env_act_day_", "")
        .replace("nutrient_", "")
        .replace("plant_img_", "plant ")
        .replace("img_", "")
        .replace("_", " ")
    )


def try_import_shap():
    try:
        import shap  # noqa: F401

        return True
    except Exception as e:
        print(f"[shap] unavailable: {type(e).__name__}: {e}", flush=True)
        return False


def shap_mean_abs_linear(pipe: Pipeline, X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Global mean|SHAP| via LinearExplainer on imputed+scaled features (ElasticNet)."""
    import shap

    Xi = pipe.named_steps["impute"].transform(X)
    Xs = pipe.named_steps["scale"].transform(Xi)
    lin = pipe.named_steps["model"]
    explainer = shap.LinearExplainer(lin, Xs)
    sv = np.asarray(explainer.shap_values(Xs))
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": np.abs(sv).mean(axis=0),
                "method": "LinearExplainer",
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def shap_mean_abs_tree(pipe: Pipeline, X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Global mean|SHAP| for tree models (XGB/GBM pipeline with imputer)."""
    import shap

    Xi = pipe.named_steps["impute"].transform(X)
    model = pipe.named_steps["model"]
    explainer = shap.Explainer(model, Xi)
    sv = explainer(Xi)
    vals = np.abs(np.asarray(sv.values)).mean(axis=0)
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": vals,
                "method": type(explainer).__name__,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def plot_importance(df: pd.DataFrame, out_png: Path, title: str, top_k: int = 15):
    sub = df.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(sub["feature"].map(nice_label), sub["mae_increase_mean"], color="#2e7d32")
    ax.set_xlabel("LODO-avg MAE increase when feature shuffled (g)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def run_nowcast(seed: int, out: Path):
    df, cols, y, days = load_nowcast()
    X = df[cols].to_numpy(dtype=float)
    model = xgb_or_gbm(seed)
    print(f"[nowcast] n={len(y)} feats={len(cols)} model=XGB/GBM", flush=True)
    imp = perm_importance_full(model, X, y, cols, seed)
    imp.to_csv(out / "nowcast_permutation_importance.csv", index=False)
    plot_importance(
        imp,
        out / "nowcast_permutation_importance.png",
        "Nowcast FW — permutation importance (full-fit explanatory)",
    )
    block = day_block_perm_importance(model, X, y, days, cols, seed)
    block.to_csv(out / "nowcast_dayblock_importance.csv", index=False)
    plot_importance(
        block,
        out / "nowcast_dayblock_importance.png",
        "Nowcast FW — day-block permutation (credits tray climate)",
    )
    fit_full_and_pdp(
        model,
        X,
        y,
        cols,
        PDP_NOWCAST,
        out / "nowcast_partial_dependence.png",
        "Nowcast FW — partial dependence",
    )
    shap_top = None
    if try_import_shap():
        m = clone(model)
        m.fit(X, y)
        try:
            shap_df = shap_mean_abs_tree(m, X, cols)
            shap_df.to_csv(out / "nowcast_shap_mean_abs.csv", index=False)
            plot_importance(
                shap_df.rename(columns={"mean_abs_shap": "mae_increase_mean"}),
                out / "nowcast_shap_mean_abs.png",
                "Nowcast FW — mean |SHAP|",
            )
            shap_top = shap_df.head(8).to_dict(orient="records")
            print("top SHAP:\n", shap_df.head(8).to_string(index=False), flush=True)
        except Exception as e:
            print(f"[shap] nowcast failed: {type(e).__name__}: {e}", flush=True)
    print("top row-shuffle:\n", imp.head(8).to_string(index=False), flush=True)
    print("top day-block:\n", block.head(8).to_string(index=False), flush=True)
    return {
        "task": "nowcast",
        "n": len(y),
        "n_features": len(cols),
        "top_row": imp.head(8).to_dict(orient="records"),
        "top_dayblock": block.head(8).to_dict(orient="records"),
        "top_shap": shap_top,
        "note": "Attribution is explanatory full-fit; predictive MAE remains LODO elsewhere.",
    }


def run_forecast(seed: int, out: Path):
    pairs, cols, y, days = load_forecast_pairs()
    X = pairs[cols].to_numpy(dtype=float)
    model = elastic(seed)
    print(f"[forecast] n={len(y)} feats={len(cols)} model=ElasticNet", flush=True)
    fw_t = pairs["fw_t"].to_numpy(dtype=float) if "fw_t" in pairs.columns else np.zeros_like(y)
    persist = float(np.mean(np.abs(y - fw_t)))
    print(f"  persist MAE={persist:.2f}g", flush=True)

    imp = perm_importance_full(model, X, y, cols, seed)
    imp.to_csv(out / "forecast_permutation_importance.csv", index=False)
    plot_importance(
        imp,
        out / "forecast_permutation_importance.png",
        "Future FW (t→t+4) — permutation importance (full-fit explanatory)",
    )
    block = day_block_perm_importance(model, X, y, days, cols, seed)
    block.to_csv(out / "forecast_dayblock_importance.csv", index=False)
    plot_importance(
        block,
        out / "forecast_dayblock_importance.png",
        "Future FW (t→t+4) — day-block permutation",
    )
    fit_full_and_pdp(
        model,
        X,
        y,
        cols,
        PDP_FORECAST,
        out / "forecast_partial_dependence.png",
        "Future FW (t→t+4) — partial dependence",
    )
    shap_top = None
    if try_import_shap():
        m = clone(model)
        m.fit(X, y)
        try:
            shap_df = shap_mean_abs_linear(m, X, cols)
            shap_df.to_csv(out / "forecast_shap_mean_abs.csv", index=False)
            # also |coef| for side-by-side paper table
            coefs = m.named_steps["model"].coef_
            cmp = (
                pd.DataFrame({"feature": cols, "abs_coef": np.abs(coefs), "coef": coefs})
                .merge(imp[["feature", "mae_increase_mean"]], on="feature")
                .merge(shap_df[["feature", "mean_abs_shap"]], on="feature")
                .sort_values("mean_abs_shap", ascending=False)
            )
            cmp.to_csv(out / "forecast_importance_compare_coef_perm_shap.csv", index=False)
            plot_importance(
                shap_df.rename(columns={"mean_abs_shap": "mae_increase_mean"}),
                out / "forecast_shap_mean_abs.png",
                "Future FW (t→t+4) — mean |SHAP| (LinearExplainer)",
            )
            shap_top = shap_df.head(8).to_dict(orient="records")
            print("top SHAP:\n", shap_df.head(8).to_string(index=False), flush=True)
        except Exception as e:
            print(f"[shap] forecast failed: {type(e).__name__}: {e}", flush=True)
    print("top row-shuffle:\n", imp.head(8).to_string(index=False), flush=True)
    print("top day-block:\n", block.head(8).to_string(index=False), flush=True)
    return {
        "task": "forecast",
        "n": len(y),
        "n_features": len(cols),
        "persist_mae_g": persist,
        "top_row": imp.head(8).to_dict(orient="records"),
        "top_dayblock": block.head(8).to_dict(orient="records"),
        "top_shap": shap_top,
        "note": "Attribution is explanatory full-fit; predictive MAE remains LODO elsewhere.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["nowcast", "forecast", "both"], default="both")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    out = Path(args.out) if args.out else OUT
    if not out.is_absolute():
        out = PROJECT / out if args.out.startswith("ml/") else OUT
    out.mkdir(parents=True, exist_ok=True)

    summary = {"seed": args.seed, "tasks": {}}
    if args.task in ("nowcast", "both"):
        summary["tasks"]["nowcast"] = run_nowcast(args.seed, out)
    if args.task in ("forecast", "both"):
        summary["tasks"]["forecast"] = run_forecast(args.seed, out)
    (out / "attribution_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}/", flush=True)


if __name__ == "__main__":
    main()

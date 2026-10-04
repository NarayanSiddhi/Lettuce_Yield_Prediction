"""
Comparison figures: crop model vs ML (image-only) vs hybrid residual ML.

Outputs under ml/figures/:
  growth_comparison_lodo.png       trajectories (like growth_trajectories.png)
  residual_comparison_lodo.png     residual bars by DAT
  mae_summary_bars.png             MAE by setup / feature set
  scatter_observed_vs_pred.png     observed vs predicted
  holdout_trial2_compact.png       train T1 -> test T2 trajectories
  mae_plant_lodo_bars.png          plant-level LODO MAE (if results exist)

Usage:
  python3 ml/plot_residual_comparisons.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent
PROJECT = ML_DIR.parent
FIG_DIR = ML_DIR / "figures"
CROP_OUT = PROJECT / "Lettuce_model" / "Python" / "outputs"
PLANT_DIR = ML_DIR / "plant"

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from train_yield_models import build_models, metrics
from train_residual_models import (
    MODELLED,
    OBSERVED,
    RESIDUAL,
    load_tray_table,
    matrix,
    select_feature_cols,
)

# Colors matching growth_trajectories.png style + ML series
C_OBS = "#d62728"
C_CROP = "#1f77b4"
C_HYBRID = "#2ca02c"
C_ML = "#ff7f0e"
C_BAND = "0.85"
C_TARGET = "0.45"


def _setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_sim(trial: int) -> pd.DataFrame:
    path = CROP_OUT / f"trial{trial}_simulation.csv"
    return pd.read_csv(path)


def load_lodo_preds(trial: str) -> pd.DataFrame:
    path = ML_DIR / f"predictions_residual_{trial}_lodo.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def plot_growth_comparison_lodo(plt) -> Path:
    """Side-by-side Trial 1 / Trial 2: observed, crop, ML image-raw, hybrid."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=True)

    for ax, trial_num, trial_key, title in (
        (axes[0], 1, "trial1", "Trial 1 — adaptive EC"),
        (axes[1], 2, "trial2", "Trial 2 — fixed EC"),
    ):
        sim = load_sim(trial_num)
        preds = load_lodo_preds(trial_key)
        preds = preds.sort_values("day")

        # steering band from crop fit if available
        fit = pd.read_csv(CROP_OUT / f"trial{trial_num}_fit.csv")
        if "target_mid_g" in fit.columns:
            ax.plot(
                fit["DAT"],
                fit["target_mid_g"],
                "--",
                color=C_TARGET,
                lw=1.2,
                label="steering mid-line",
            )

        ax.plot(
            sim["DAT"],
            sim["fresh_weight_g"],
            "-",
            color=C_CROP,
            lw=2.0,
            label="crop model (physics)",
        )
        ax.plot(
            preds["day"],
            preds["pred_hybrid_g"],
            "-s",
            color=C_HYBRID,
            lw=1.8,
            ms=5,
            label=f"hybrid residual ({preds['best_hybrid_model'].iloc[0]})",
        )
        ax.plot(
            preds["day"],
            preds["pred_image_raw_g"],
            "-^",
            color=C_ML,
            lw=1.4,
            ms=5,
            label=f"ML image-only ({preds['best_image_raw_model'].iloc[0]})",
        )
        ax.plot(
            preds["day"],
            preds["observed_g"],
            "o",
            color=C_OBS,
            ms=7,
            label="observed median",
            zorder=5,
        )

        # MAE annotations
        crop_mae = float(np.mean(np.abs(preds["observed_g"] - preds["modelled_g"])))
        hyb_mae = float(np.mean(np.abs(preds["abs_err_hybrid_g"])))
        ml_mae = float(np.mean(np.abs(preds["abs_err_image_raw_g"])))
        ax.set_title(
            f"{title}\nLODO MAE — crop {crop_mae:.1f} g | "
            f"hybrid {hyb_mae:.1f} g | ML-only {ml_mae:.1f} g",
            fontsize=10,
        )
        ax.set_xlabel("Days after transplant")
        ax.set_xticks(preds["day"].tolist())
        ax.grid(alpha=0.3)
        if trial_num == 1:
            ax.axvline(12, color="#ff7f0e", ls="-.", lw=1.0, alpha=0.7)
            ax.annotate(
                "PPFD↑",
                (12, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 230),
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                color="#ff7f0e",
            )

    axes[0].set_ylabel("Fresh weight [g plant$^{-1}$]")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Crop model vs ML — leave-one-harvest-day-out predictions",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out = FIG_DIR / "growth_comparison_lodo.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_residual_bars(plt) -> Path:
    """Residual (obs - pred) by DAT for crop / hybrid / ML."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), sharey=True)
    width = 1.1

    for ax, trial_key, title in (
        (axes[0], "trial1", "Trial 1"),
        (axes[1], "trial2", "Trial 2"),
    ):
        p = load_lodo_preds(trial_key).sort_values("day")
        days = p["day"].to_numpy()
        crop_res = (p["observed_g"] - p["modelled_g"]).to_numpy()
        hyb_res = (p["observed_g"] - p["pred_hybrid_g"]).to_numpy()
        ml_res = (p["observed_g"] - p["pred_image_raw_g"]).to_numpy()

        ax.bar(days - width, crop_res, width=width, color=C_CROP, label="crop residual")
        ax.bar(days, hyb_res, width=width, color=C_HYBRID, label="hybrid residual")
        ax.bar(days + width, ml_res, width=width, color=C_ML, label="ML-only residual")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("Days after transplant")
        ax.set_xticks(days.tolist())
        ax.grid(alpha=0.3, axis="y")

    axes[0].set_ylabel("Error (observed − predicted) [g]")
    axes[0].legend(fontsize=8)
    fig.suptitle("Where each model is wrong (LODO)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "residual_comparison_lodo.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_scatter(plt) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharex=True, sharey=True)
    series = [
        ("modelled_g", "Crop model", C_CROP),
        ("pred_hybrid_g", "Hybrid residual ML", C_HYBRID),
        ("pred_image_raw_g", "ML image-only", C_ML),
    ]
    frames = [load_lodo_preds("trial1"), load_lodo_preds("trial2")]
    all_p = pd.concat(frames, ignore_index=True)
    lim = [0, max(240, float(all_p["observed_g"].max()) * 1.05)]

    for ax, (col, title, color) in zip(axes, series):
        for trial, marker in (("trial1", "o"), ("trial2", "s")):
            sub = all_p[all_p["trial"] == trial]
            ax.scatter(
                sub["observed_g"],
                sub[col],
                c=color if trial == "trial1" else "none",
                edgecolors=color,
                marker=marker,
                s=55,
                label=trial,
                linewidths=1.5,
            )
        ax.plot(lim, lim, "k--", lw=1, alpha=0.5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("Observed [g]")
        ax.grid(alpha=0.3)
        mae = float(np.mean(np.abs(all_p["observed_g"] - all_p[col])))
        ax.text(0.05, 0.95, f"MAE={mae:.1f} g", transform=ax.transAxes, va="top", fontsize=9)

    axes[0].set_ylabel("Predicted [g]")
    axes[0].legend(fontsize=8)
    fig.suptitle("Observed vs predicted (both trials, LODO)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "scatter_observed_vs_pred.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_mae_bars(plt) -> Path:
    """Bar chart of best MAE per setup from result CSVs."""
    rows = []
    for trial in ("trial1", "trial2"):
        for variant in ("image", "compact", "sensors"):
            path = ML_DIR / f"results_residual_{trial}_{variant}_lodo.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            for setup_prefix, label in (
                ("crop_only", "crop"),
                (f"{variant}_hybrid", "hybrid"),
                (f"{variant}_raw", "ML raw"),
            ):
                sub = df[df["setup"] == setup_prefix]
                if sub.empty:
                    continue
                best = sub.loc[sub["mae_g"].astype(float).idxmin()]
                rows.append(
                    {
                        "trial": trial,
                        "variant": variant,
                        "setup": label,
                        "mae": float(best["mae_g"]),
                        "model": best["model"],
                    }
                )

    if not rows:
        # fall back to older combined image LODO files
        for trial in ("trial1", "trial2"):
            path = ML_DIR / f"results_residual_{trial}_lodo.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            for setup, label in (
                ("crop_only", "crop"),
                ("hybrid_residual", "hybrid"),
                ("image_raw", "ML raw"),
            ):
                sub = df[df["setup"] == setup]
                if sub.empty:
                    continue
                best = sub.loc[sub["mae_g"].astype(float).idxmin()]
                rows.append(
                    {
                        "trial": trial,
                        "variant": "image",
                        "setup": label,
                        "mae": float(best["mae_g"]),
                        "model": best["model"],
                    }
                )

    res = pd.DataFrame(rows)
    variants = list(dict.fromkeys(res["variant"].tolist()))
    fig, axes = plt.subplots(1, len(variants), figsize=(4.8 * len(variants), 4.2), squeeze=False)

    setup_order = ["crop", "hybrid", "ML raw"]
    colors = {"crop": C_CROP, "hybrid": C_HYBRID, "ML raw": C_ML}
    x = np.arange(2)
    width = 0.25

    for ax, variant in zip(axes[0], variants):
        for i, setup in enumerate(setup_order):
            vals = []
            for trial in ("trial1", "trial2"):
                hit = res[(res["variant"] == variant) & (res["setup"] == setup) & (res["trial"] == trial)]
                vals.append(float(hit["mae"].iloc[0]) if len(hit) else np.nan)
            ax.bar(x + (i - 1) * width, vals, width=width, color=colors[setup], label=setup)
        ax.set_xticks(x)
        ax.set_xticklabels(["Trial 1", "Trial 2"])
        ax.set_ylabel("MAE [g]" if variant == variants[0] else "")
        ax.set_title(f"Feature set: {variant}")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    fig.suptitle("Best LODO MAE by approach", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "mae_summary_bars.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def holdout_compact_predictions(seed: int = 42) -> pd.DataFrame:
    """Train compact residual / raw models on Trial 1; predict Trial 2."""
    df1 = load_tray_table("trial1")
    df2 = load_tray_table("trial2")
    feat_cols = select_feature_cols(df1, "compact", plant=False)
    X1, X2 = matrix(df1, feat_cols), matrix(df2, feat_cols)
    y1 = pd.to_numeric(df1[OBSERVED], errors="coerce").to_numpy(dtype=float)
    r1 = pd.to_numeric(df1[RESIDUAL], errors="coerce").to_numpy(dtype=float)
    y2 = pd.to_numeric(df2[OBSERVED], errors="coerce").to_numpy(dtype=float)
    m2 = pd.to_numeric(df2[MODELLED], errors="coerce").to_numpy(dtype=float)

    best_raw = None
    best_hyb = None
    for name, model in build_models(random_state=seed):
        model.fit(X1, y1)
        pred_raw = model.predict(X2).astype(float)
        mae_raw = float(np.mean(np.abs(y2 - pred_raw)))
        if best_raw is None or mae_raw < best_raw[1]:
            best_raw = (name, mae_raw, pred_raw)

        model.fit(X1, r1)
        pred_hyb = m2 + model.predict(X2).astype(float)
        mae_hyb = float(np.mean(np.abs(y2 - pred_hyb)))
        if best_hyb is None or mae_hyb < best_hyb[1]:
            best_hyb = (name, mae_hyb, pred_hyb)

    out = df2[["sample_id", "day", "date"]].copy()
    out["observed_g"] = y2
    out["modelled_g"] = m2
    out["pred_image_raw_g"] = best_raw[2]
    out["pred_hybrid_g"] = best_hyb[2]
    out["best_image_raw_model"] = best_raw[0]
    out["best_hybrid_model"] = best_hyb[0]
    out["abs_err_crop_only_g"] = np.abs(y2 - m2)
    out["abs_err_image_raw_g"] = np.abs(y2 - best_raw[2])
    out["abs_err_hybrid_g"] = np.abs(y2 - best_hyb[2])
    return out


def plot_holdout_trial2(plt) -> Path:
    preds = holdout_compact_predictions()
    preds.to_csv(ML_DIR / "predictions_residual_holdout_trial2_compact.csv", index=False)
    sim = load_sim(2)
    fit = pd.read_csv(CROP_OUT / "trial2_fit.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))

    # Left: trajectories
    ax = axes[0]
    if "target_mid_g" in fit.columns:
        ax.plot(fit["DAT"], fit["target_mid_g"], "--", color=C_TARGET, lw=1.2, label="steering mid-line")
    ax.plot(sim["DAT"], sim["fresh_weight_g"], "-", color=C_CROP, lw=2.0, label="crop model")
    ax.plot(
        preds["day"],
        preds["pred_hybrid_g"],
        "-s",
        color=C_HYBRID,
        lw=1.8,
        ms=5,
        label=f"hybrid ({preds['best_hybrid_model'].iloc[0]})",
    )
    ax.plot(
        preds["day"],
        preds["pred_image_raw_g"],
        "-^",
        color=C_ML,
        lw=1.4,
        ms=5,
        label=f"ML raw ({preds['best_image_raw_model'].iloc[0]})",
    )
    ax.plot(preds["day"], preds["observed_g"], "o", color=C_OBS, ms=7, label="observed", zorder=5)
    crop_mae = float(preds["abs_err_crop_only_g"].mean())
    hyb_mae = float(preds["abs_err_hybrid_g"].mean())
    ml_mae = float(preds["abs_err_image_raw_g"].mean())
    ax.set_title(
        f"Hold-out Trial 2 (train Trial 1, compact features)\n"
        f"MAE — crop {crop_mae:.1f} g | hybrid {hyb_mae:.1f} g | ML-only {ml_mae:.1f} g"
    )
    ax.set_xlabel("Days after transplant")
    ax.set_ylabel("Fresh weight [g plant$^{-1}$]")
    ax.set_xticks(preds["day"].tolist())
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # Right: per-day absolute error
    ax = axes[1]
    days = preds["day"].to_numpy()
    width = 1.1
    ax.bar(days - width, preds["abs_err_crop_only_g"], width=width, color=C_CROP, label="crop")
    ax.bar(days, preds["abs_err_hybrid_g"], width=width, color=C_HYBRID, label="hybrid")
    ax.bar(days + width, preds["abs_err_image_raw_g"], width=width, color=C_ML, label="ML raw")
    ax.set_xlabel("Days after transplant")
    ax.set_ylabel("|error| [g]")
    ax.set_title("Absolute error by harvest day")
    ax.set_xticks(days.tolist())
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    fig.suptitle("Generalization test: train Trial 1 → predict Trial 2", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "holdout_trial2_compact.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_plant_mae_bars(plt) -> Path | None:
    path = PLANT_DIR / "results_residual_plant_all.csv"
    if not path.exists():
        # stitch from per-variant LODO files
        frames = []
        for variant in ("image", "compact", "sensors"):
            p = PLANT_DIR / f"results_residual_plant_{variant}_lodo.csv"
            if p.exists():
                frames.append(pd.read_csv(p))
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path)
        df = df[df["split"] == "lodo"]

    rows = []
    for variant in sorted(df["variant"].dropna().unique()):
        sub = df[df["variant"] == variant]
        for setup_key, label in (
            ("crop_only", "crop"),
            (f"{variant}_hybrid", "hybrid"),
            (f"{variant}_raw", "ML raw"),
        ):
            hit = sub[sub["setup"] == setup_key]
            if hit.empty:
                continue
            best = hit.loc[hit["mae_g"].astype(float).idxmin()]
            rows.append({"variant": variant, "setup": label, "mae": float(best["mae_g"])})

    if not rows:
        return None

    res = pd.DataFrame(rows)
    variants = list(dict.fromkeys(res["variant"].tolist()))
    setup_order = ["crop", "hybrid", "ML raw"]
    colors = {"crop": C_CROP, "hybrid": C_HYBRID, "ML raw": C_ML}

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(variants))
    width = 0.25
    for i, setup in enumerate(setup_order):
        vals = []
        for v in variants:
            hit = res[(res["variant"] == v) & (res["setup"] == setup)]
            vals.append(float(hit["mae"].iloc[0]) if len(hit) else np.nan)
        ax.bar(x + (i - 1) * width, vals, width=width, color=colors[setup], label=setup)

    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("Plant-level MAE [g]")
    ax.set_title("Plant-level LODO (79 rows) — best MAE by feature set")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "mae_plant_lodo_bars.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    plt = _setup_mpl()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    written.append(plot_growth_comparison_lodo(plt))
    print(f"Wrote {written[-1]}")

    written.append(plot_residual_bars(plt))
    print(f"Wrote {written[-1]}")

    written.append(plot_scatter(plt))
    print(f"Wrote {written[-1]}")

    written.append(plot_mae_bars(plt))
    print(f"Wrote {written[-1]}")

    written.append(plot_holdout_trial2(plt))
    print(f"Wrote {written[-1]}")
    print(f"Also wrote {ML_DIR / 'predictions_residual_holdout_trial2_compact.csv'}")

    plant = plot_plant_mae_bars(plt)
    if plant:
        written.append(plant)
        print(f"Wrote {plant}")

    print(f"\n{len(written)} figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()

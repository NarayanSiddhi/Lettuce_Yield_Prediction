"""Run the lettuce crop model over one or both grow-tent trials.

Examples
--------
    python run_lettuce_trial.py                    # both trials, calibrated
    python run_lettuce_trial.py --trial 1          # just Trial 1
    python run_lettuce_trial.py --no-calibrate     # uncalibrated baseline
    python run_lettuce_trial.py --canopy-efficiency 1.8
    python run_lettuce_trial.py --temp-basis photoperiod
    python run_lettuce_trial.py --ec-response      # enable the EC modifier
    python run_lettuce_trial.py --no-plot

Outputs land in ./outputs/:
    trial{N}_simulation.csv    full daily state and fluxes
    trial{N}_fit.csv           checkpoint comparison, including the residual
    growth_trajectories.png    model vs observed vs steering band
    carbon_balance.png         the fluxes behind the trajectory
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lettuce_model import (TRIAL_META, LeafOptics, calibrate_canopy_efficiency,
                           fit_report, lettuce_params, load_drivers,
                           load_observed, simulate_trial, target_band)
from lettuce_model.calibrate import fit_metrics
from lettuce_model.targets import deviation_from_band, gompertz

OUT = Path(__file__).resolve().parent / "outputs"


def run_one(trial: int, args, optics: LeafOptics):
    meta = TRIAL_META[trial]
    print(f"\n{'=' * 72}\n{meta['label']}  ({meta['transplant']} -> "
          f"{meta['harvest']})\n{'=' * 72}")
    print(f"  EC strategy : {meta['ec_strategy']}")
    print(f"  Light       : {meta['light']}")

    drivers = load_drivers(trial)
    observed = load_observed(trial)

    params = lettuce_params(temp_basis=args.temp_basis,
                            ec_response=args.ec_response)
    if args.growing_area_m2:
        params = params.with_(growing_area_m2=args.growing_area_m2)
    if args.dry_matter_fraction:
        params = params.with_(dry_matter_fraction=args.dry_matter_fraction)

    print(f"\n  Planting density : {params.planting_density:.1f} plants/m2 "
          f"({params.n_plants} plants on {params.growing_area_m2:.2f} m2)")
    print(f"  Dry matter frac  : {params.dry_matter_fraction:.3f}")
    print(f"  Photosynthesis T : {params.temp_basis} mean")
    print(f"  Leaf optics      : {optics}")

    # ---- calibration -------------------------------------------------
    if args.canopy_efficiency is not None:
        params = params.with_(canopy_efficiency=args.canopy_efficiency)
        cal_info = {"canopy_efficiency": args.canopy_efficiency,
                    "source": "command line"}
    elif args.no_calibrate:
        cal_info = {"canopy_efficiency": params.canopy_efficiency,
                    "source": "uncalibrated default (1.0)"}
    else:
        ce, cal_info = calibrate_canopy_efficiency(drivers, observed, params,
                                                   optics=optics)
        cal_info["source"] = "least squares in log fresh weight"
        params = params.with_(canopy_efficiency=ce)
        if cal_info["at_bound"]:
            print("\n  WARNING: the fit sat on a search bound. The model "
                  "cannot reach the observed biomass by scaling canopy "
                  "assimilation alone -- treat the value below as a lower/"
                  "upper limit, not an estimate.")

    print(f"  Canopy efficiency: {params.canopy_efficiency:.3f} "
          f"({cal_info['source']})")

    # ---- simulate ----------------------------------------------------
    sim = simulate_trial(drivers, params, optics=optics)
    report = fit_report(sim, observed)
    metrics = fit_metrics(report)

    print("\n  Model vs observed median fresh weight [g/plant]:")
    print(report.to_string())
    print(f"\n  MAE {metrics['MAE_g']:.1f} g | RMSE {metrics['RMSE_g']:.1f} g "
          f"| MAPE {metrics['MAPE_pct']:.1f}% | R2 {metrics['R2']:.3f} "
          f"| bias {metrics['bias_g']:+.1f} g")
    print(f"  Harvest (DAT 28): observed {metrics['final_observed_g']:.1f} g, "
          f"modelled {metrics['final_modelled_g']:.1f} g")

    if sim.get("buffer_clamped", pd.Series(dtype=bool)).any():
        n = int(sim["buffer_clamped"].sum())
        print(f"  NOTE: assimilate buffer hit zero on {n} day(s) -- the crop "
              "was carbon-limited then.")

    # ---- where the trial sat relative to its own steering band --------
    dev = deviation_from_band(observed["DAT"], observed["observed_median_g"])
    n_in = int(dev["in_band"].sum())
    print(f"\n  Observed crop was inside the steering band at "
          f"{n_in}/{len(dev)} checkpoints.")

    OUT.mkdir(parents=True, exist_ok=True)
    sim.to_csv(OUT / f"trial{trial}_simulation.csv")
    report.to_csv(OUT / f"trial{trial}_fit.csv")

    return dict(trial=trial, params=params, sim=sim, report=report,
                metrics=metrics, observed=observed, drivers=drivers,
                calibration=cal_info)


def make_plots(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping plots "
              "(pip install matplotlib)")
        return

    band = target_band()

    # ---- Figure 1: growth trajectories --------------------------------
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 5.0),
                             sharey=True, squeeze=False)
    for ax, r in zip(axes[0], results):
        meta = TRIAL_META[r["trial"]]
        ax.fill_between(band.index, band["target_low_g"], band["target_high_g"],
                        color="0.85", label="steering target band")
        ax.plot(band.index, band["target_mid_g"], "--", color="0.45", lw=1.2,
                label="target mid-line (expolinear)")
        t = np.arange(0, 29)
        ax.plot(t, gompertz(t), ":", color="0.55", lw=1.1,
                label="Gompertz as written in Ch.3")
        ax.plot(r["sim"].index, r["sim"]["fresh_weight_g"], "-", color="#1f77b4",
                lw=2.0, label="crop model")
        ax.plot(r["observed"]["DAT"], r["observed"]["observed_median_g"], "o",
                color="#d62728", ms=7, label="observed median")

        if r["trial"] == 1:
            ax.axvline(12, color="#ff7f0e", ls="-.", lw=1.2)
            ax.annotate("PPFD 200 -> 445", (12, 250), rotation=90,
                        va="top", ha="right", fontsize=8, color="#ff7f0e")

        ax.set_title(f"{meta['label']}\ncanopy efficiency = "
                     f"{r['params'].canopy_efficiency:.2f}, "
                     f"MAE = {r['metrics']['MAE_g']:.1f} g", fontsize=10)
        ax.set_xlabel("Days after transplant")
        ax.grid(alpha=0.3)
        ax.set_xticks(band.index)
    axes[0][0].set_ylabel("Fresh weight [g plant$^{-1}$]")
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "growth_trajectories.png", dpi=160)
    plt.close(fig)

    # ---- Figure 2: what drove it --------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for r in results:
        meta = TRIAL_META[r["trial"]]
        lab = meta["short"]
        s, d = r["sim"], r["drivers"]
        axes[0][0].plot(d.index, d["DLI_mol_m2_d"], label=lab)
        axes[0][1].plot(s.index, s["f_light"], label=lab)
        axes[1][0].plot(s.index[1:], s["P_gross_gCH2O_m2_d"].iloc[1:],
                        label=f"{lab} gross P")
        axes[1][0].plot(s.index[1:], s["R_maint_gCH2O_m2_d"].iloc[1:], "--",
                        label=f"{lab} maintenance R")
        axes[1][1].plot(d.index, d["EC_mS_cm"], label=lab)

    axes[0][0].set_ylabel("DLI [mol m$^{-2}$ d$^{-1}$]")
    axes[0][0].set_title("Light supplied")
    axes[0][1].set_ylabel("intercepted fraction [-]")
    axes[0][1].set_title("Canopy light interception")
    axes[1][0].set_ylabel("g CH$_2$O m$^{-2}$ d$^{-1}$")
    axes[1][0].set_title("Carbon balance")
    axes[1][1].set_ylabel("EC [mS cm$^{-1}$]")
    axes[1][1].set_title("Nutrient solution EC (measured)")
    for ax in axes.ravel():
        ax.set_xlabel("Days after transplant")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "carbon_balance.png", dpi=160)
    plt.close(fig)

    print(f"\nWrote {OUT / 'growth_trajectories.png'}")
    print(f"Wrote {OUT / 'carbon_balance.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial", type=int, choices=[1, 2], default=None,
                    help="run one trial (default: both)")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="run at canopy_efficiency = 1.0 instead of fitting it")
    ap.add_argument("--canopy-efficiency", type=float, default=None,
                    help="force a specific canopy efficiency")
    ap.add_argument("--temp-basis", choices=["daily", "photoperiod"],
                    default="daily",
                    help="temperature/humidity average driving photosynthesis")
    ap.add_argument("--ec-response", action="store_true",
                    help="enable the empirical EC growth modifier "
                         "(off by default; see params.py)")
    ap.add_argument("--growing-area-m2", type=float, default=None,
                    help="override the bench floor area [m2]")
    ap.add_argument("--dry-matter-fraction", type=float, default=None,
                    help="override the shoot dry matter fraction [-]")
    ap.add_argument("--spectrum-column", type=int, default=9,
                    help="1-based column of relative_pfd.txt (MATLAB "
                         "convention); 9 = broad white LED")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    optics = LeafOptics.from_files(spectrum_column=args.spectrum_column)
    trials = [args.trial] if args.trial else [1, 2]
    results = [run_one(t, args, optics) for t in trials]

    if len(results) == 2:
        a, b = results
        print(f"\n{'=' * 72}\nCROSS-TRIAL COMPARISON\n{'=' * 72}")
        print(f"  Observed harvest difference : "
              f"{a['metrics']['final_observed_g'] - b['metrics']['final_observed_g']:+.1f} g "
              f"(adaptive minus fixed)")
        print(f"  Modelled harvest difference : "
              f"{a['metrics']['final_modelled_g'] - b['metrics']['final_modelled_g']:+.1f} g")
        print("\n  The model sees only light, temperature, CO2 and humidity.")
        print("  Whatever of the observed difference it does NOT reproduce is")
        print("  the part attributable to EC steering, cultivar variation, or")
        print("  anything else outside the model -- and that residual is what")
        print("  the canopy-image analysis is being asked to explain.")

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        f"trial{r['trial']}": {
            "label": TRIAL_META[r["trial"]]["label"],
            "calibration": r["calibration"],
            "metrics": r["metrics"],
        } for r in results
    }
    with open(OUT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nWrote {OUT / 'summary.json'}")

    if not args.no_plot:
        make_plots(results)


if __name__ == "__main__":
    main()

"""Fit the single calibration coefficient, and report how well it did.

WHY ONLY ONE FREE PARAMETER
---------------------------
There are eight fresh-weight checkpoints per trial, and they are not eight
independent observations -- they are eight views of the same ten plants, four
days apart, so the effective sample size is smaller still.  A model with three
or four free parameters will fit that beautifully and predict nothing.  So
exactly one coefficient moves, `canopy_efficiency`, and everything else stays
at its literature or measured value.  If the fit is bad, that is a result
about the model, which is more useful than a good fit obtained by tuning.

WHAT IS FITTED
--------------
Least squares in LOG fresh weight, not in grams.  Growth spans 4.7 g to 220 g
over the cycle, so a linear-space fit is decided almost entirely by the last
two checkpoints and ignores whether the model got the establishment phase
right.  Log space weights proportional error equally at every checkpoint,
which is the right question here: the project's downstream use is
early-warning detection of plants departing from trajectory, and that needs
the early points to be right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .growth import simulate_trial
from .params import CropParams
from .spectral import LeafOptics


def _predicted_at(sim: pd.DataFrame, dats) -> np.ndarray:
    """Model fresh weight at the observation days."""
    return np.array([float(sim.loc[int(d), "fresh_weight_g"]) for d in dats])


def calibrate_canopy_efficiency(drivers: pd.DataFrame,
                                observed: pd.DataFrame,
                                params: CropParams,
                                optics: LeafOptics | None = None,
                                bounds: tuple[float, float] = (0.2, 6.0),
                                ) -> tuple[float, dict]:
    """Fit `canopy_efficiency` to a trial's observed fresh-weight checkpoints.

    Parameters
    ----------
    drivers : DataFrame       daily drivers, indexed by DAT
    observed : DataFrame      columns DAT and observed_median_g
    params : CropParams       everything except canopy_efficiency is held fixed
    bounds : (lo, hi)         search interval.  The upper bound of 6 is
                              deliberately generous so that an implausible fit
                              shows up as a large number rather than being
                              silently clipped at a respectable-looking one.

    Returns
    -------
    (best_value, info) where info carries the objective and the search status.
    """
    dats = observed["DAT"].to_numpy()
    obs = observed["observed_median_g"].to_numpy(dtype=float)

    # DAT 0 is an identity by construction (the model is initialised from it),
    # so including it would reward the fit for nothing.  Drop it.
    keep = dats > 0
    dats, obs = dats[keep], obs[keep]

    def objective(ce: float) -> float:
        sim = simulate_trial(drivers, params.with_(canopy_efficiency=float(ce)),
                             optics=optics)
        pred = _predicted_at(sim, dats)
        pred = np.maximum(pred, 1e-6)
        return float(np.mean((np.log(pred) - np.log(obs)) ** 2))

    res = minimize_scalar(objective, bounds=bounds, method="bounded",
                          options={"xatol": 1e-4})

    info = {
        "canopy_efficiency": float(res.x),
        "objective_log_mse": float(res.fun),
        "converged": bool(res.success),
        "n_checkpoints": int(len(dats)),
        "bounds": bounds,
        "at_bound": bool(abs(res.x - bounds[0]) < 1e-3
                         or abs(res.x - bounds[1]) < 1e-3),
    }
    return float(res.x), info


def fit_report(sim: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Checkpoint-by-checkpoint comparison of model against observation.

    The `residual_g` column is the quantity this whole project is built
    around: it is what a mechanistic model driven by light, temperature, CO2
    and humidity CANNOT explain, and therefore what canopy image features and
    EC history are being asked to predict.
    """
    rows = []
    for _, r in observed.iterrows():
        d = int(r["DAT"])
        if d not in sim.index:
            continue
        pred = float(sim.loc[d, "fresh_weight_g"])
        obs = float(r["observed_median_g"])
        rows.append({
            "DAT": d,
            "observed_g": obs,
            "modelled_g": round(pred, 1),
            "residual_g": round(obs - pred, 1),
            "residual_pct": round(100.0 * (obs - pred) / max(obs, 1e-9), 1),
            "target_mid_g": float(r.get("target_mid_g", np.nan)),
        })
    return pd.DataFrame(rows).set_index("DAT")


def fit_metrics(report: pd.DataFrame, skip_dat0: bool = True) -> dict:
    """Summary error statistics for a fit report."""
    df = report[report.index > 0] if skip_dat0 else report
    resid = df["residual_g"].to_numpy(dtype=float)
    obs = df["observed_g"].to_numpy(dtype=float)
    pred = df["modelled_g"].to_numpy(dtype=float)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    return {
        "n": int(len(df)),
        "MAE_g": float(np.mean(np.abs(resid))),
        "RMSE_g": float(np.sqrt(np.mean(resid ** 2))),
        "MAPE_pct": float(np.mean(np.abs(resid / np.maximum(obs, 1e-9))) * 100),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "bias_g": float(np.mean(resid)),
        "final_observed_g": float(obs[-1]),
        "final_modelled_g": float(pred[-1]),
    }

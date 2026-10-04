"""The growth-target trajectory the trials were actually steered against.

This is NOT the crop model.  It is the reference curve the supervisory logic
compared median fresh weight against every four days when deciding whether to
raise EC.  It is reproduced here for two reasons: the crop model has to be
plotted against the same band to be comparable with the thesis figures, and
the band's real shape is not quite what the thesis describes.

WHAT THE THESIS SAYS vs. WHAT THE STEERING TOOL DID
---------------------------------------------------
Thesis Ch.3 describes a Gompertz curve, W(t) = A*exp(-exp(b - k*t)) with
A = 227 g, b = 3.2, k = 0.20, and a flat +/-10% tolerance band.

The band actually used, from `Data collection/Trial1_FreshWeight_EC_tracker.xlsx`
(the operator's `Targets` sheet), is different in two ways:

  1. The mid-line is exponential then linear, not Gompertz.  From DAT 0 to 12
     it grows by a constant factor of 2.52 every 4 days (relative growth rate
     0.2312 d-1); from DAT 16 to 28 it grows by a constant 10.68 g d-1.  Both
     limbs are exact to the last digit in the sheet.  That is an expolinear
     growth curve (Goudriaan & Monteith 1990), the standard form for a crop
     that goes from an open to a closed canopy -- which is arguably a better
     choice for lettuce than Gompertz, but it is not what Ch.3 documents.

  2. The tolerance band narrows as the crop matures: +/-15% for DAT 0-12,
     +/-10% for DAT 16-24, +/-7% at harvest.  The thesis reports a single
     +/-10%.  The narrowing is sensible -- a 15% miss at 3.5 g is noise, a
     15% miss at 227 g is a failed crop -- but it means "within the band" is
     a stricter test late than the thesis text implies.

`target_band()` returns the tabulated values, i.e. what the trial was really
run against.  `gompertz()` returns the curve Ch.3 describes, for comparison.
Both are exposed so a figure can show the discrepancy rather than pick a side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The operator's tabulated target band, verbatim from
# Data collection/FreshWeight_checkpoints_both_trials.xlsx
# (identical for both trials).
_TABLE = {
    #  DAT:  (low_g,  mid_g,  high_g)
    0:  (3.0,   3.5,   4.0),
    4:  (7.5,   8.8,   10.1),
    8:  (18.9,  22.2,  25.5),
    12: (47.6,  56.0,  64.4),
    16: (88.9,  98.8,  108.6),
    20: (127.4, 141.5, 155.7),
    24: (165.8, 184.3, 202.7),
    28: (211.1, 227.0, 242.9),
}

# Fitted description of the tabulated mid-line, exact to the sheet's precision.
EXPO_RGR_PER_DAY = np.log(2.52) / 4.0     # 0.2312 d-1, DAT 0-12
LINEAR_RATE_G_PER_DAY = 10.68             # g plant-1 d-1, DAT 16-28
TRANSITION_DAT = 12


def target_band(dat=None) -> pd.DataFrame:
    """Steering target band, one row per checkpoint.

    Parameters
    ----------
    dat : iterable of int, optional
        Restrict to these checkpoints.  Default: all eight.

    Returns
    -------
    DataFrame indexed by DAT with target_low_g, target_mid_g, target_high_g
    and tolerance_pct (the half-width actually applied at that checkpoint).
    """
    rows = []
    for d, (lo, mid, hi) in sorted(_TABLE.items()):
        rows.append({
            "DAT": d,
            "target_low_g": lo,
            "target_mid_g": mid,
            "target_high_g": hi,
            # Recovered from the table rather than assumed, so the narrowing
            # is visible in the output instead of buried in a constant.
            "tolerance_pct": round(100.0 * (hi - mid) / mid, 1),
        })
    out = pd.DataFrame(rows).set_index("DAT")
    if dat is not None:
        out = out.loc[list(dat)]
    return out


def expolinear(t):
    """The tabulated mid-line, as a continuous function of DAT.

    Exponential at a constant relative growth rate until canopy closure,
    linear at a constant absolute rate afterwards -- the two-phase shape a
    crop shows when it moves from light-limited-by-leaf-area to
    light-limited-by-incident-radiation.
    """
    t = np.asarray(t, dtype=float)
    w0 = _TABLE[0][1]                                        # 3.5 g
    exp_phase = w0 * np.exp(EXPO_RGR_PER_DAY * t)
    w_at_transition = w0 * np.exp(EXPO_RGR_PER_DAY * TRANSITION_DAT)
    lin_phase = w_at_transition + LINEAR_RATE_G_PER_DAY * (t - TRANSITION_DAT)
    return np.where(t <= TRANSITION_DAT, exp_phase, lin_phase)


def gompertz(t, A: float = 227.0, b: float = 3.2, k: float = 0.20):
    """The Gompertz curve described in thesis Chapter 3.

    W(t) = A * exp(-exp(b - k*t)).

    Kept for comparison only.  It is NOT the curve the operator's spreadsheet
    tabulates: at DAT 16 it gives 83.5 g against the sheet's 98.8 g, and at
    DAT 28 it gives 207.3 g against 227.0 g.  Cite carefully.
    """
    t = np.asarray(t, dtype=float)
    return A * np.exp(-np.exp(b - k * t))


def deviation_from_band(dat, fresh_weight_g) -> pd.DataFrame:
    """Where a fresh-weight series sits relative to the steering band.

    Returns a frame with the relative deviation from the mid-line and a flag
    for whether each checkpoint fell inside the band -- the quantity the
    supervisory EC logic keyed on.
    """
    band = target_band()
    rows = []
    for d, w in zip(np.atleast_1d(dat), np.atleast_1d(fresh_weight_g)):
        if int(d) not in band.index:
            continue
        b = band.loc[int(d)]
        rows.append({
            "DAT": int(d),
            "fresh_weight_g": float(w),
            "target_mid_g": b["target_mid_g"],
            "deviation_pct": 100.0 * (w - b["target_mid_g"]) / b["target_mid_g"],
            "in_band": bool(b["target_low_g"] <= w <= b["target_high_g"]),
        })
    return pd.DataFrame(rows).set_index("DAT")

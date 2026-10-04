"""Load the prepared daily driver tables and the observed fresh-weight data.

`prepare_drivers.py` writes these files from the raw Mycodo exports; this
module only reads them, so the raw-export parsing lives in exactly one place.
"""

from __future__ import annotations

import os

import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "data")


TRIAL_META = {
    1: dict(
        label="Trial 1 - corrective adaptive EC steering",
        short="adaptive",
        transplant="2025-10-14",
        harvest="2025-11-10",
        ec_strategy="1.2-1.4 mS/cm at transplant, stepped up at 4-day "
                    "checkpoints when median FW fell below the target band "
                    "(+0.2 mS/cm below band, -0.1 mS/cm after two checkpoints "
                    "above it); measured daily mean reached 2.25 mS/cm",
        light="~200 umol m-2 s-1 for DAT 0-11, raised to ~445 at DAT 12 "
              "and held there",
        observed_final_g=219.8,
    ),
    2: dict(
        label="Trial 2 - preventive fixed EC",
        short="fixed",
        transplant="2026-01-20",
        harvest="2026-02-16",
        ec_strategy="held at 1.4-1.6 mS/cm for the whole cycle; growth "
                    "checkpoints used for evaluation only, never for "
                    "intervention",
        light="~445 umol m-2 s-1 constant from transplant",
        observed_final_g=179.8,
    ),
}


def load_drivers(trial: int, data_dir: str | None = None) -> pd.DataFrame:
    """Daily driver table for a trial, indexed by day after transplant."""
    data_dir = data_dir or _DATA_DIR
    path = os.path.join(data_dir, f"trial{trial}_daily_drivers.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python prepare_drivers.py` first.")
    df = pd.read_csv(path, index_col="DAT")
    df.attrs.update(TRIAL_META.get(trial, {}))
    return df


def load_observed(trial: int | None = None,
                  data_dir: str | None = None) -> pd.DataFrame:
    """Observed median net fresh weight and the steering target band.

    Columns: trial, DAT, target_low_g, target_mid_g, target_high_g,
    observed_median_g.  Weights are NET of the 25 g substrate + net-pot
    correction the thesis applies (Section 3.5.4), so they are directly
    comparable with the model's shoot fresh weight.
    """
    data_dir = data_dir or _DATA_DIR
    path = os.path.join(data_dir, "observed_fresh_weight.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python prepare_drivers.py` first.")
    obs = pd.read_csv(path)
    if trial is not None:
        obs = obs[obs["trial"] == trial].reset_index(drop=True)
    return obs

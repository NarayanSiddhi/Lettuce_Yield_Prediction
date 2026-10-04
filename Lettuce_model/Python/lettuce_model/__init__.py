"""Mechanistic lettuce crop model for the Cal Poly Pomona grow-tent trials.

A two-state (structural / non-structural dry matter) daily growth model in the
style of Van Henten (1994), driven by a Farquhar-von Caemmerer-Berry (FvCB)
leaf photosynthesis module.  Adapted from the lettuce crop model that runs
inside the CEAC greenhouse web tool (`C:\\CEAC_proposal_modeling`).

Public entry points
-------------------
    CropParams, TRIAL_PARAMS      -- parameter sets (params.py)
    LeafOptics                    -- spectral absorptance / quantum yield
    photosynthesis                -- FvCB leaf gas exchange
    simulate_day, simulate_trial  -- the growth model itself
    load_drivers, load_observed   -- trial data I/O
    target_band                   -- the steering trajectory used in the trials
    calibrate_canopy_efficiency   -- one-parameter fit to observed biomass
"""

from .params import CropParams, TRIAL_PARAMS, lettuce_params
from .spectral import LeafOptics
from .photosynthesis import photosynthesis
from .growth import simulate_day, simulate_trial, initial_state
from .drivers import load_drivers, load_observed, TRIAL_META
from .targets import target_band, gompertz
from .calibrate import calibrate_canopy_efficiency, fit_report

__all__ = [
    "CropParams", "TRIAL_PARAMS", "lettuce_params",
    "LeafOptics", "photosynthesis",
    "simulate_day", "simulate_trial", "initial_state",
    "load_drivers", "load_observed", "TRIAL_META",
    "target_band", "gompertz",
    "calibrate_canopy_efficiency", "fit_report",
]

__version__ = "1.0.0"

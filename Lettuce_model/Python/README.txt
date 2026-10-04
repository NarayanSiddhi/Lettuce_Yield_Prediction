# Lettuce crop model — Python

A mechanistic daily growth model for the hydroponic lettuce grown in the
Cal Poly Pomona automated grow tent, adapted from the lettuce crop model
inside the CEAC greenhouse web tool (`C:\CEAC_proposal_modeling`).

The MATLAB version in `../MATLAB/` implements the same equations and
reproduces these numbers exactly. Use whichever you prefer.

---

## 1. Install

You need Python 3.9 or newer. Check what you have:

```bash
python --version
```

### Option A — a virtual environment (recommended)

Keeps these packages separate from anything else on the machine, so nothing
you install here can break another project.

**Windows (PowerShell):**

```powershell
cd "C:\Users\eravishankar\OneDrive - Cal Poly Pomona\Lettuce_Siddhi_Sai_collab\Lettuce_model\Python"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to run the activate script ("running scripts is
disabled on this system"), allow it for your own account once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**Windows (Command Prompt):** same, but activate with
`.venv\Scripts\activate.bat`.

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You will see `(.venv)` at the start of your prompt when it is active. Run
`deactivate` to leave it. You need to activate it again in every new terminal.

### Option B — install into your existing Python

Fine if you do not mind the packages being global:

```bash
pip install -r requirements.txt
```

### Option C — conda

```bash
conda create -n lettuce python=3.11
conda activate lettuce
conda install -c conda-forge numpy pandas scipy matplotlib openpyxl
```

### What each package is for

| Package | Why it is needed |
|---|---|
| `numpy` | array maths throughout the model |
| `pandas` | driver tables and result frames |
| `scipy` | one function, `minimize_scalar`, used only by the calibration step |
| `matplotlib` | figures — **optional**; the model runs and writes CSVs without it |
| `openpyxl` | reads the `.xlsx` Mycodo exports in `prepare_drivers.py` |

### Check it worked

```bash
python -c "import numpy, pandas, scipy, matplotlib, openpyxl; print('all good')"
```

---

## 2. Run

```bash
# 1. Build the daily driver tables from the raw Mycodo exports (once).
python prepare_drivers.py

# 2. Run the model on both trials, calibrate, and write figures.
python run_lettuce_trial.py
```

Everything lands in `outputs/`:

| File | Contents |
|---|---|
| `trial{N}_simulation.csv` | full daily state and all carbon fluxes |
| `trial{N}_fit.csv` | model vs observed at each checkpoint, **including the residual** |
| `growth_trajectories.png` | model vs observed vs the steering target band |
| `carbon_balance.png` | the light, interception and flux series behind the trajectory |
| `summary.json` | calibration values and error metrics |

### Useful variations

```bash
python run_lettuce_trial.py --trial 1              # one trial only
python run_lettuce_trial.py --no-calibrate         # uncalibrated baseline
python run_lettuce_trial.py --canopy-efficiency 1.4
python run_lettuce_trial.py --temp-basis photoperiod
python run_lettuce_trial.py --ec-response          # enable the EC modifier
python run_lettuce_trial.py --growing-area-m2 1.86 # once the bench is measured
python run_lettuce_trial.py --no-plot
python run_lettuce_trial.py --help
```

---

## 3. Using it as a library

```python
from lettuce_model import (lettuce_params, load_drivers, load_observed,
                           simulate_trial, calibrate_canopy_efficiency,
                           fit_report)

drivers  = load_drivers(1)
observed = load_observed(1)
params   = lettuce_params()

ce, info = calibrate_canopy_efficiency(drivers, observed, params)
sim      = simulate_trial(drivers, params.with_(canopy_efficiency=ce))

print(fit_report(sim, observed))     # residual_g is the column that matters
```

`simulate_trial` returns a DataFrame indexed by day after transplant, with
the two state variables, fresh weight per plant, LAI, light interception,
and the daily carbon fluxes.

---

## 4. What the model does

Two state variables carry the crop:

- **`W_struct`** — structural dry matter (cell walls). Determines leaf area
  and harvestable weight.
- **`W_buffer`** — the non-structural pool of soluble sugars and starch.

Photosynthesis pays into the buffer; maintenance respiration draws from it
around the clock; structural growth draws from it at a rate that saturates as
the buffer fills. The feedback from `W_struct` back into photosynthesis
through leaf area is what produces the sigmoid growth curve.

Each day:

1. **Optics** (`spectral.py`) — incident PPFD → absorbed photons and
   electron-driving flux, weighted by the LED spectrum, leaf absorptance and
   the McCree quantum-yield curve.
2. **Leaf gas exchange** (`photosynthesis.py`) — FvCB with Leuning stomatal
   coupling. The smoothed minimum of Rubisco-, RuBP- and TPU-limited rates,
   then a stomatal loop that makes assimilation respond to VPD.
3. **Canopy scaling** (`growth.py`) — Beer–Lambert interception on LAI, then
   integrate over the 16 h photoperiod.
4. **Carbon balance** (`growth.py`) — maintenance respiration over 24 h,
   structural growth at Van Henten's saturating rate, synthesis overhead.
5. **Fresh weight** — shoot dry matter ÷ dry-matter fraction ÷ planting density.

Drivers per day: air temperature, relative humidity, CO₂, PPFD, pressure.
Optionally EC, which is **ignored by default** — see below.

### The model has no nutrient module, on purpose

It predicts growth from light, temperature, CO₂ and humidity alone. That is
what makes it useful here: **the model-minus-observed residual is a clean
target** for the canopy-image and EC analysis to explain. Switching on the
optional EC response (`--ec-response`) moves part of the EC signal *into* the
model and *out of* the residual, which is a different and much weaker
experiment.

---

## 5. Fitted results

One free parameter, `canopy_efficiency`, fitted per trial by least squares in
log fresh weight against the eight 4-day checkpoints. Everything else is at a
literature or measured value.

| | Trial 1 (adaptive) | Trial 2 (fixed) |
|---|---|---|
| Canopy efficiency | 1.42 | 0.99 |
| MAE | 9.4 g | 10.5 g |
| RMSE | 10.9 g | 13.3 g |
| R² | 0.976 | 0.945 |
| Harvest, observed | 219.8 g | 179.8 g |
| Harvest, modelled | 201.4 g | 150.9 g |

Both fitted values are physically plausible (1–1.5 for a shallow canopy in a
mylar-lined tent), which is worth something: a fit that only worked at
canopy efficiency 4 would be telling you the model was wrong somewhere else.

**Read the residual pattern, do not tune it away.** Both trials show the same
shape: the model runs ahead of the crop at DAT 16–20 and behind it at
harvest. The measured crop had a pronounced mid-cycle stall (relative growth
rate fell to 0.08 d⁻¹ around DAT 8–12) and then a late surge the model does
not produce. That is a real, repeatable feature of both trials and it is
exactly the kind of thing the image analysis exists to detect early.

One caution on the DAT 4 points: both trials jump to 13–18 g at DAT 4 and
then barely move to DAT 8. That is not a plausible growth pattern, and it is
the largest single residual in both fits. Check the DAT 4 weighing protocol —
in particular whether the 25 g substrate correction was applied consistently
to plants that small.

---

## 6. What to check before quoting absolute yields

Two numbers are assumptions, not measurements, and both scale the output
close to linearly:

1. **`growing_area_m2 = 1.49`** — inferred from the apparent 4 ft × 4 ft tent
   footprint. Measure the NFT bench. A 10% error here is a 10% error in
   predicted head weight.
2. **`dry_matter_fraction = 0.045`** — the trials recorded fresh weight only.
   Oven-dry a few heads at the next harvest and this stops being a guess.

Neither affects the *shape* of the trajectory or the residual pattern, so the
model is already usable for the residual analysis. They matter for absolute
yield claims.

---

## 7. File map

```
Python/
├── README.txt                  this file
├── requirements.txt
├── prepare_drivers.py         raw Mycodo exports -> ../data/*.csv
├── run_lettuce_trial.py       command-line runner
├── outputs/                   results (regenerated on every run)
└── lettuce_model/
    ├── params.py              every parameter, with units and provenance
    ├── spectral.py            LED spectrum x leaf optics -> two scalars
    ├── photosynthesis.py      FvCB + Leuning
    ├── growth.py              the daily carbon balance
    ├── drivers.py             data loading, trial metadata
    ├── targets.py             the steering target band the trials used
    └── calibrate.py           the one-parameter fit and its error metrics
```

See `../README.txt` for how this relates to the MATLAB version and to the
original CEAC model, including the list of corrections made along the way.

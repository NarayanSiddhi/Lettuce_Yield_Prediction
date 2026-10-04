# Lettuce crop model

A mechanistic daily growth model for the hydroponic lettuce grown in the
Cal Poly Pomona automated grow tent (Trials 1 and 2). Two implementations —
MATLAB and Python — of the same equations, producing identical numbers.

Adapted from the lettuce crop model inside the CEAC greenhouse web tool at
`C:\CEAC_proposal_modeling\Lettuce_model`.

---

## Why the project needs this

The image-based yield-prediction work in `../Initial plan/` reached a
sensible conclusion: rather than train a model to predict biomass directly
from environmental sensors — which cannot separate cause from controller,
since EC was *adjusted in response to* poor growth — use a mechanistic crop
model to predict expected growth from the physics, and then ask what the
images and the nutrient history explain about the part it gets wrong.

That gives the machine learning a defined biological role:

```
crop model  ->  expected biomass at DAT t   (from light, temperature, CO2, humidity)
measurement ->  observed biomass at DAT t   (every 4 days)
                --------------------------
residual    =   what the physics cannot explain
                        |
                        v
        can canopy image features + EC history predict it?
```

This folder supplies the first box. `outputs/trial{N}_fit.csv` supplies the
residual.

---

## Layout

```
Lettuce_model/
├── README.txt            this file
├── data/                driver tables, observations, spectral data
│   ├── trial1_daily_drivers.csv
│   ├── trial2_daily_drivers.csv
│   ├── observed_fresh_weight.csv
│   ├── relative_pfd.txt
│   └── lettuce_spectrum.txt
├── MATLAB/              MATLAB implementation + README
└── Python/              Python implementation + README (install instructions)
```

Start with `Python/README.txt` (it has the install steps) or
`MATLAB/README.txt`.

**The driver tables are built by `Python/prepare_drivers.py`**, which parses
the raw Mycodo exports. Both implementations read the same CSVs, so any
difference between their outputs is a difference in the *model*, not in the
data preparation. Run it once before either implementation:

```bash
cd Python && python prepare_drivers.py
```

---

## What the model is

Two state variables carry the crop, in the style of Van Henten (1994):

- **structural dry matter** — cell walls; determines leaf area and
  harvestable weight
- **non-structural dry matter** — the soluble sugar and starch buffer

Photosynthesis pays into the buffer, maintenance respiration draws from it
around the clock, and structural growth draws from it at a rate that
saturates as the buffer fills. The feedback from structure back to
photosynthesis through leaf area produces the sigmoid growth curve.

Leaf photosynthesis is Farquhar–von Caemmerer–Berry with Leuning stomatal
coupling — the same module the CEAC web tool runs — so the model responds to
CO₂ and to the tent's VPD control, not only to light and temperature.

Daily drivers: air temperature, relative humidity, CO₂, PPFD, pressure.

### It deliberately has no nutrient module

The model predicts growth from light, temperature, CO₂ and humidity alone.
That is the point: **the residual is then a clean target** for the image and
EC analysis. An optional empirical EC response exists and is **off by
default**; switching it on moves EC signal into the model and out of the
residual, which is a different and much weaker experiment.

---

## Results

One free parameter (`canopy_efficiency`), fitted per trial by least squares in
log fresh weight against the eight 4-day checkpoints. Everything else sits at
a literature or measured value.

| | Trial 1 (adaptive EC) | Trial 2 (fixed EC) |
|---|---|---|
| Canopy efficiency (fitted) | 1.42 | 0.99 |
| MAE | 9.4 g | 10.5 g |
| RMSE | 10.9 g | 13.3 g |
| R² | 0.976 | 0.945 |
| Harvest observed | 219.8 g | 179.8 g |
| Harvest modelled | 201.4 g | 150.9 g |

Observed harvest difference **+40.0 g**; modelled **+50.5 g**. So the light
schedule alone — Trial 1's step from 200 to 445 µmol m⁻² s⁻¹ at DAT 12 versus
Trial 2's constant 445 — is more than sufficient to account for the whole
harvest gap between the two trials, before EC is invoked at all. That is
consistent with what the thesis itself concedes in Section 3.4.4: the light
timing difference confounds the EC comparison, and observed differences are
system-level outcomes rather than isolated effects of nutrient steering.

**The residual pattern is the finding, not an error to tune away.** Both
trials show the same shape: the model runs ahead of the crop at DAT 16–20 and
behind it at harvest. The measured crop stalled mid-cycle (relative growth
rate down to 0.08 d⁻¹ around DAT 8–12) then surged late. That is repeatable
across both trials and is exactly what an early-warning image model should be
trying to detect.

---

## Known limitations — read before quoting numbers

1. **Bench area is an estimate.** `growing_area_m2 = 1.49` is inferred from
   the apparent 4 ft × 4 ft tent footprint, not measured. It sets planting
   density, which scales biomass per m² to per plant, so a 10% error here is
   a 10% error in predicted head weight. Measure the bench.

2. **Dry matter fraction is an assumption.** The trials recorded fresh weight
   only, so `dry_matter_fraction = 0.045` is a literature midpoint. Oven-dry
   a few heads at the next harvest.

Neither affects the *shape* of the trajectory or the residual pattern, so the
model is already usable for the residual analysis.

3. **PPFD was never logged.** There was no PAR sensor in the control loop —
   light was set by manual fixture dimming and mounting height. The 200 and
   445 µmol m⁻² s⁻¹ values come from the thesis protocol and are treated as
   exact step functions. Since light is the model's dominant driver, an error
   here propagates directly. A logged quantum sensor would be the single
   highest-value addition to the next trial.

4. **Daily time step.** Photosynthesis is evaluated once per day at the mean
   conditions over the light period. That is nearly exact for a step-function
   LED schedule; do not reuse it for sunlight-driven runs without checking.

5. **Two trials, run sequentially, are not replicates.** Everything here is
   descriptive. It cannot support a causal claim about EC steering, and it is
   not intended to.

---

## Data provenance

| What | Source |
|---|---|
| Trial 1 environment (T, RH, CO₂, pressure) | `Data collection/Trial1_environment_log.csv` |
| Trial 1 solution (EC, temperature, volume) | `Data collection/Trial1_nutrient_log.csv` |
| Trial 2 environment and solution | `Data collection/Trial2_Mycodo_raw_export.xlsx` |
| Fresh weight checkpoints, target band | `Data collection/FreshWeight_checkpoints_both_trials.xlsx` |
| Trial 1 per-plant weights, EC decisions | `Data collection/Trial1_FreshWeight_EC_tracker.xlsx` |
| Light schedule, protocol, setpoints | `Sam_thesis/` — thesis Ch. 3, Tables 3.1–3.3 |
| Spectral data, FvCB module, base parameters | `C:\CEAC_proposal_modeling\Lettuce_model` |

### Transplant dates

Pinned by the first logger record of each trial, in both cases an evening
timestamp on the day the seedlings went in (Trial 1 camera 2025-10-14
19:20:54; Trial 2 Mycodo 2026-01-19 19:20:36). Adding the 28-day cycle lands
harvest exactly on the last day of data in both trials — the consistency
check that settles the DAT numbering. One consequence: the two camera folders
are offset by a day, with Trial 1 images spanning DAT 0–27 and Trial 2 images
DAT 1–28.

### Sensor QC applied

`prepare_drivers.py` drops readings outside physical plausibility windows
before averaging. The one that matters: the Atlas EZO-CO₂ NDIR circuit emits
sub-ppm values during warm-up and after I²C read errors — **1358 of 5549
Trial 2 readings are below 300 ppm**. Averaging those in would drag the daily
mean CO₂ down by roughly 120 ppm.

---

## A note on the steering target band

The thesis (Ch. 3) describes the growth target as a Gompertz curve,
*W = 227·exp(−exp(3.2 − 0.20t))*, with a flat ±10% tolerance band. The band
actually tabulated in the operator's `Lettuce_FW_EC_Tracker` sheet — the one
the trial was really steered against — is neither:

- Its mid-line is **expolinear**, not Gompertz: growing by a constant factor
  of 2.52 every 4 days to DAT 12 (RGR 0.2312 d⁻¹), then by a constant
  10.68 g d⁻¹ from DAT 16. Both limbs are exact to the sheet's last digit.
- Its tolerance **narrows** with maturity: ±15% for DAT 0–12, ±10% for
  DAT 16–24, ±7% at harvest.

The Gompertz curve as written gives 83.5 g at DAT 16 against the sheet's
98.8 g, and 207.3 g at DAT 28 against 227.0 g. Both curves are plotted in
`growth_trajectories.png` so the difference is visible rather than buried.
Worth reconciling before the thesis text is finalised.

---

## References

The papers behind the model are in `../Literature/`, with an annotated index
in `../Literature/README.txt`. The two that matter most here:

- **Van Henten, E.J. (1994).** Validation of a dynamic lettuce growth model
  for greenhouse climate control. *Agricultural Systems* 45:55–72. — the
  two-state structure, and every growth and respiration coefficient.
- **Farquhar, von Caemmerer & Berry (1980).** A biochemical model of
  photosynthetic CO₂ assimilation in leaves of C3 species. *Planta*
  149:78–90. — the leaf photosynthesis module.

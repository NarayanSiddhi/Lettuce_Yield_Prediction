# Lettuce crop model — MATLAB

Same model as `../Python/`, same equations, same numbers. Tested on
MATLAB R2024b. No toolboxes required beyond base MATLAB (`fminbnd` and
`readtable` are both core).

---

## Run it

**Prerequisite, once:** the daily driver tables in `../data/` are built from
the raw Mycodo exports by the Python script. Parsing the raw exports in one
place only means MATLAB and Python are driven by *identical* numbers, so any
difference in their output is a difference in the model rather than in the
data prep.

```
cd ..\Python
python prepare_drivers.py
```

Then, in MATLAB:

```matlab
cd '...\Lettuce_model\MATLAB'
run_lettuce_trial
```

Options are the block at the top of `run_lettuce_trial.m`:

```matlab
TRIALS            = [1 2];      % which trials to run
CALIBRATE         = true;       % fit canopyEfficiency to observed biomass
CANOPY_EFFICIENCY = [];         % or force a value
TEMP_BASIS        = 'daily';    % 'daily' or 'photoperiod'
EC_RESPONSE       = false;      % enable the empirical EC modifier
SPECTRUM_COLUMN   = 9;          % 9 = broad white LED panel
GROWING_AREA_M2   = [];         % override once the bench is measured
DRY_MATTER_FRAC   = [];
MAKE_PLOTS        = true;
```

Results land in `outputs/`.

---

## Using it directly

```matlab
p        = lettuceParams();
optics   = leafOptics();
drivers  = loadTrialData('drivers',  1);
observed = loadTrialData('observed', 1);

[ce, info] = calibrateCanopyEfficiency(drivers, observed, p, optics);
p.canopyEfficiency = ce;

sim = simulateLettuceTrial(drivers, p, optics);
plot(sim.DAT, sim.freshWeightG)
```

---

## Files

| File | What it does |
|---|---|
| `run_lettuce_trial.m` | driver script: options, calibration, reporting, figures |
| `lettuceParams.m` | every parameter, with units and provenance |
| `leafOptics.m` | LED spectrum × leaf optics → two scalars |
| `photosynthesisFvCB.m` | FvCB leaf gas exchange with Leuning stomata |
| `cropGeometry.m` | LAI, light interception, fresh weight, initial state, EC factor |
| `simulateLettuceDay.m` | one day of the carbon balance |
| `simulateLettuceTrial.m` | loop over a trial |
| `calibrateCanopyEfficiency.m` | the one-parameter fit |
| `loadTrialData.m` | driver tables, observations, target band, trial metadata |

---

## Relationship to the original CEAC MATLAB

This is a rewrite of `C:\CEAC_proposal_modeling\Lettuce_model`, not a copy.
The physiology, the FvCB module and the Van Henten parameter values all come
from there. Four things were changed, and each is flagged with a `CORRECTION`
comment at the line it affects.

### 1. State variable naming

The original calls its two states `Ws` and `Wg` and labels them "shoot" and
"root". The equations do not use them that way — fresh weight and leaf area
are both computed from `Wg`, which the comment calls root, and the assimilate
balance is applied to `Ws`. The mapping that the equations actually follow is:

| Original | Here | What it is |
|---|---|---|
| `Ws` | `Wbuffer` | non-structural assimilate pool (Van Henten `X_nsdw`) |
| `Wg` | `Wstruct` | structural dry weight (Van Henten `X_sdw`) |

### 2. Spectral units (`leafOptics.m` vs `parabs_batch.m`)

`parabs_batch.m` treats a column of `relative_pfd.txt` as a spectral
irradiance in W m⁻² nm⁻¹ and converts it to photons with *E = hc/λ*. But that
file is a relative **photon** flux distribution — its columns integrate to ~1
in photon units. Passing photons through a watt-to-photon conversion inflates
the result by a factor of about 4.2, so the original reports an absorbed PPFD
roughly **three times the incident PPFD**, which is not physically possible.
Here the spectrum is normalised in photon units over the 400–700 nm band, so
the absorbed fraction is bounded by 1 by construction.

### 3. Which pool structural growth is proportional to

Original: `r_gr = c_gr_max * Ws/(Wg+Ws) * fT ; dWg = r_gr * Ws` — growth
proportional to the **buffer**. Van Henten (1994) has growth proportional to
the **structure**, throttled by a saturation function of buffer filling. The
denominators coincide when `c_gamma = 1`; the pool the rate multiplies does
not.

### 4. Leaf area, and why the default differs from Van Henten

Original: `f_light = 1 - exp(-0.7 * c_lar * Wg)` with `c_lar = 0.075`. This
drops the shoot fraction `(1 - c_tau)` and uses an extinction coefficient of
0.7 where Van Henten specifies 0.9.

More consequentially, `c_lar = 0.075` gives **LAI 9.2 at a 220 g head** and
98% light interception by DAT 20. Neither is true of this crop — lettuce at
200–250 g carries LAI 4–5, and the DAT 20 camera frames show wide gaps
between plants with the white gutters plainly visible. In the model that
premature saturation turns growth linear around DAT 20, whereas the measured
crop stayed close to exponential to harvest.

The default here (`larModel = 'declining'`) instead builds LAI from a
measured lettuce specific leaf area falling from 300 to 150 cm² g⁻¹ as leaves
thicken, giving LAI ≈ 4.6 at 227 g. **Neither SLA value was tuned against the
fresh-weight data** — this is a correction on physical grounds, not a fit.
Van Henten's original form is still available with
`p.larModel = 'constant'`, and it is instructive to run it and watch the
trajectory shape go wrong.

Also worth knowing: the original CEAC masterfile sets `c_gr_max = 1e-6`,
five times below the 5×10⁻⁶ s⁻¹ Van Henten's paper reports. That is why the
web tool takes ~70 days to reach a 250 g head against a 30–45 day commercial
norm. The published value is used here.

---

## Results

Identical to the Python implementation to the printed precision, which is the
cross-check that both are doing what they claim:

| | Trial 1 (adaptive) | Trial 2 (fixed) |
|---|---|---|
| Canopy efficiency (fitted) | 1.416 | 0.992 |
| MAE | 9.4 g | 10.5 g |
| RMSE | 10.9 g | 13.3 g |
| R² | 0.976 | 0.945 |
| Harvest, observed | 219.8 g | 179.8 g |
| Harvest, modelled | 201.4 g | 150.9 g |

See `../README.txt` and `../Python/README.txt` for what the residuals mean and
what to check before quoting absolute yields.

# Dataset Audit — Lettuce Yield Prediction

**Date:** 2026-09-14  
**Purpose:** Quantify what the dataset can support before locking Aim 1 / Aim 2 architecture (roadmap §1, §17, §20).  
**Status:** Inventory complete. Action items listed at end.

---

## 1. Executive decision (from this audit)

| Factor | Finding |
|--------|---------|
| Independent growth cycles | **2 only** (Trial 1, Trial 2) |
| Image sequences | **Moderate** (821 frames; 42 day-level t→t+4 pairs) |
| Fresh-weight labels | **Sparse** (8 checkpoints × 2 trials; plant-level only Trial 1) |
| Sensors | Trial 1 rich (+ actuators); Trial 2 sensors only; **PPFD never logged** |
| Critical bugs | `sensor_features_daily_trial2.csv` is a **byte-identical copy of Trial 1** |

**Roadmap §17 classification:** *moderate images, few independent trials.*

**Implication for the paper:**
- **Primary:** phenotype extraction + multimodal temporal FW/ΔFW forecasting + ablations + honest T1↔T2 generalization.
- **Secondary:** conditioned future-image generation (pretrained + PEFT), judged by phenotype/biomass consistency — not a from-scratch diffusion paper.
- **Do not** center novelty on “SegFormer replaces FastSAM” alone (FLAsH 2026 already uses fine-tuned FastSAM for lettuce).

---

## 2. Trials / cycles

| Trial | Protocol | Calendar span | Duration |
|-------|----------|---------------|----------|
| Trial 1 | Adaptive EC steering | 2025-10-14 → 2025-11-11 (images through 2025-11-10) | 28 DAT |
| Trial 2 | Preventive fixed EC | 2026-01-19 → 2026-02-16 (images from 2026-01-20) | 28 DAT |

Source: `Data collection/README.txt`.

---

## 3. Images

| | Trial 1 | Trial 2 | Total |
|--|--------:|--------:|------:|
| Indexed usable frames | 430 | 391 | **821** |
| Days with ≥1 image | 28 | 25 | — |
| Day range (images) | 2025-10-14 … 2025-11-10 | 2026-01-20 … 2026-02-16 | — |
| Camera gaps | Missing harvest day 2025-11-11 | Missing **2026-02-05, 02-06, 02-07** | — |
| Typical cadence | ~hourly 05:00–20:00 | same | — |
| FastSAM segmented dirs on disk | 430 | 391 | 821 |

Index: `segmentation/timelapse_index.csv`.  
Segmented outputs: `Timelapse_growtent_segmented/` (~10 GB).

**Known issue:** camera FOV differs between trials (close vs wide) — complicates cross-trial visual transfer.

### t → t+4 image pairing

| Level | Trial 1 | Trial 2 | Total |
|-------|--------:|--------:|------:|
| Calendar days with images on both t and t+4 | 24 | 18 | **42** |
| Same-hour frame pairs (approx from prior audit) | 362 | 270 | **632** |
| Cup-level pairs (archived daily features / forecast tables) | 43 | 78 | **121** |

---

## 4. Fresh-weight labels

### Tray / median (both trials)

File: `Lettuce_model/data/observed_fresh_weight.csv` — **16 rows**  
Checkpoints: DAT **0, 4, 8, 12, 16, 20, 24, 28**.

Also: `Data collection/FreshWeight_checkpoints_both_trials.xlsx`.

| DAT | T1 median (g) | T2 median (g) |
|----:|-------------:|-------------:|
| 0 | 4.7 | 4.7 |
| 4 | 13.1 | 17.8 |
| 8 | 19.8 | 20.8 |
| 12 | 27.3 | 26.3 |
| 16 | 42.3 | 41.8 |
| 20 | 80.8 | 70.8 |
| 24 | 139.3 | 119.3 |
| 28 | 219.8 | 179.8 |

**Caveat:** README notes DAT 4 weights as biologically implausible in both trials — flag in paper / sensitivity analysis.

### Plant-level (Trial 1 only)

- `Data collection/Trial1_FreshWeight_EC_tracker.xlsx` → cleaned `Lettuce_model/data/trial1_plant_level_fresh_weight.csv`
- **10 plants × 8 days = 80** records
- **Trial 2: no plant-level FW**

### ML label bugs to fix before publishable results

- Some `ml/checkpoint_trial2*.csv` / combined tables use **Trial 1 protocol-shifted FW**, not `observed_fresh_weight.csv` Trial 2 medians (e.g. DAT28 ML ~218 vs true **179.8**).
- Prefer `observed_fresh_weight.csv` as canonical tray labels.

---

## 5. Sensors, environment, actuators

### Trial 1 — full stack

| Source | Content |
|--------|---------|
| `Data collection/Trial1_environment_log.csv` | CO₂, air T, RH, pressure; actuators: LIGHT, AC, EXHAUST, HUMIDIFIER, HEATER |
| `Data collection/Trial1_nutrient_log.csv` | EC, solution T, volume, flow; pH Up/Down, PUMP, Nutrient A/B/C |
| `Data collection/Trial1_Mycodo_raw_export.xlsx` | Adds VPD, dewpoint, pH, TDS, etc. |
| `sensor_features_daily_trial1.csv` | 29 days × ~110 engineered day/night + actuator features |
| `Lettuce_model/data/trial1_daily_drivers.csv` | Daily crop-model drivers |

### Trial 2 — sensors only

| Source | Content |
|--------|---------|
| `Data collection/Trial2_Mycodo_raw_export.xlsx` | pH, CO₂, EC, T, RH, VPD (~7.9k rows; 2026-01-19 → 2026-02-16) |
| `Lettuce_model/data/trial2_daily_drivers.csv` | Daily drivers (usable) |
| Actuator durations | **Missing** |
| `sensor_features_daily_trial2.csv` | **INVALID** — identical to Trial 1 (same dates 2025-10-14…, same bytes) |

### Variable checklist

| Variable | Trial 1 | Trial 2 |
|----------|---------|---------|
| EC | yes | yes |
| pH | yes | yes (Mycodo) |
| Air temp / RH / VPD / CO₂ | yes | yes (CO₂ noisy: many &lt;300 ppm) |
| PPFD | **protocol constant only** (not measured) | same |
| Actuators / dosing | **yes** | **no** |

---

## 6. Existing ML tables (do not treat as audit-clean)

| File pattern | Approx size | Notes |
|--------------|-------------|-------|
| `ml/checkpoint_trial1*.csv` | 8 rows | Tray checkpoints; LOO used historically |
| `ml/checkpoint_trial2*.csv` | 8 rows | Sensors/labels unreliable until fixed |
| `ml/checkpoint_combined.csv` | 16 rows | Thin for multimodal claims |
| `ml/plant/` | 79 rows (T1) | Plant-level; only subset have cup images |
| Pix2Pix/ControlNet pair CSVs | ~121 cup pairs | Generative baselines already run (poor visual metrics) |

---

## 7. Splits / generalization feasibility

| Design | Feasible? | Limitation |
|--------|-----------|------------|
| Within-trial time split (early → late) | Yes | Standard; use for primary metrics |
| Leave-one-trial-out (train T1 / test T2) | Technically yes | **N=2 cycles only** — report as hard test, not definitive |
| Plant-level LOTO | No | No T2 plant weights |
| Random image split | Avoid for paper claims | Same trajectory leakage |

Freeze identical splits across all models once master timeline is built.

---

## 8. Segmentation annotation status (supporting Aim 1)

| Item | Status |
|------|--------|
| FastSAM full pass | Done (821 frames) |
| Research draft masks (1/day) | **53** under `segmentation/research/dataset/masks_draft/` |
| Human-corrected GT | Pending (`masks_gt/`) |
| Role in paper | Benchmark FastSAM vs U-Net/SegFormer; **not** sole novelty |

---

## 9. Action items from this audit (ordered)

1. ~~**Rebuild Trial 2 daily sensor features** from `Trial2_Mycodo_raw_export.xlsx`~~ **DONE 2026-09-14** (`sensor_features_daily_trial2.csv`; bogus copy archived under `research/archive_phase0_20260914/`).
2. ~~**Correct ML checkpoint labels** for Trial 2 using `observed_fresh_weight.csv`~~ **DONE 2026-09-14** (patched `ml/checkpoint_trial2*.csv` + combined).
3. ~~**Build master timeline table**~~ **DONE** → `research/tables/master_timeline.csv`.
4. ~~**Freeze split protocol**~~ **DONE** → `research/SPLIT_PROTOCOL.md`.
5. Continue GT mask correction in parallel (human).
6. Rebuild multimodal checkpoint feature windows with real T2 sensors (next engineering step).
7. Only then run E2/E3 multimodal experiments; Aim 2 generation after predictive validity.

---

## 10. One-line paper-facing summary

*Two 28-day greenhouse cycles with dense imagery and sensors but sparse biomass labels; multimodal t+4 growth forecasting is supported, while large generative models are a secondary, consistency-checked contribution—not the primary claim.*

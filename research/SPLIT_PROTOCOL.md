# Split Protocol (frozen 2026-09-14)

Identical rules for all Aim 1 / Aim 2 experiments. Do not mix schemes across models.

## Units of analysis

| Level | ID | Notes |
|-------|-----|-------|
| Tray / trial-day | `trial`, `DAT`, `date` | Primary for FW medians |
| Plant (Trial 1 only) | `plant_id` + `DAT` | No T2 plant weights |
| Cup / image | `cup_id` + `date` + frame | For phenotype & generation pairs |

Canonical labels: `research/tables/fw_tray_canonical.csv`  
Master timeline: `research/tables/master_timeline.csv`

## Required splits

### A. Within-trial time split (primary reporting)
- Sort by `DAT`.
- Train: early checkpoints / days; Val: middle; Test: latest held-out days.
- Suggested tray checkpoints (8 days): **Train DAT ∈ {0,4,8,12}**, **Val DAT ∈ {16,20}**, **Test DAT ∈ {24,28}**.
- Never put the same `DAT` of the same trial in two splits.

### B. Leave-one-trial-out (hard generalization)
- Fold 1: train Trial 1 → test Trial 2.
- Fold 2: train Trial 2 → test Trial 1 (image/sensor only where labels allow).
- Report both; emphasize limitations (**N = 2 cycles**).

### C. Forbidden
- Random row shuffles that mix adjacent days of the same growth curve into train and test.
- Using future sensor windows that include times after the prediction time `t`.
- Claiming yellow % = calcium deficiency.

## Feature leakage rules

For predicting FW(t) or FW(t+4) / generating image(t+4):
- Sensors/actuators: only history **≤ t** (for t+4 target: history in `[t-H, t]`).
- Images/phenotypes: only ≤ t (unless the task is reconstruction of the same day).
- Window length default: **4 days** ending at t (matches harvest cadence).

## Seeds & reporting
- Fix `seed=42` unless comparing seeds; then report mean ± SD over ≥3 seeds when feasible.
- Same split files for every model comparison.
- Untouched final test until freeze.

## Files to use after Phase 0
- Sensors T2: `sensor_features_daily_trial2.csv` (rebuilt from Mycodo; **not** the archived invalid copy).
- FW: `research/tables/fw_tray_canonical.csv` / patched `ml/checkpoint_trial2*.csv`.

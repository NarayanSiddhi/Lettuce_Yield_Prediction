"""
Join crop-model checkpoint fits with FastSAM + sensor checkpoint features.

Reads:
  Lettuce_model/Python/outputs/trial{N}_fit.csv
  master_checkpoint_dataset_trial{N}.csv

Writes:
  ml/checkpoint_residual_trial{N}.csv
  ml/checkpoint_residual_combined.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
CROP_FIT_DIR = PROJECT_ROOT / "Lettuce_model" / "Python" / "outputs"
ML_DIR = PROJECT_ROOT / "ml"

RESIDUAL_COLS = (
    "observed_g",
    "modelled_g",
    "residual_g",
    "residual_pct",
    "crop_target_mid_g",
)


def load_crop_fit(trial: str) -> pd.DataFrame:
    path = CROP_FIT_DIR / f"{trial}_fit.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    fit = pd.read_csv(path)
    fit = fit.rename(columns={"DAT": "day", "target_mid_g": "crop_target_mid_g"})
    fit["day"] = fit["day"].astype(int)
    return fit


def build_trial(trial: str) -> pd.DataFrame:
    checkpoint_path = PROJECT_ROOT / f"master_checkpoint_dataset_{trial}.csv"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    chk = pd.read_csv(checkpoint_path)
    fit = load_crop_fit(trial)

    merged = chk.merge(fit, on="day", how="left", validate="one_to_one")
    missing = merged["residual_g"].isna().sum()
    if missing:
        raise ValueError(f"{trial}: {missing} checkpoint rows missing crop-model fit")

    merged["target_median_fw_g"] = merged["observed_g"]
    merged["target_residual_g"] = merged["residual_g"]
    return merged


def build_plant_trial1() -> pd.DataFrame:
    """Trial 1 plant-level rows with crop-model tray prediction attached."""
    plant_path = ML_DIR / "plant" / "plant_checkpoint_trial1_with_meta.csv"
    if not plant_path.exists():
        raise FileNotFoundError(
            f"{plant_path} missing. Run: python3 export_plant_ml_datasets.py"
        )
    plant = pd.read_csv(plant_path)
    fit = load_crop_fit("trial1")
    merged = plant.merge(fit[["day", "modelled_g", "residual_g"]], on="day", how="left")
    merged = merged.rename(columns={"residual_g": "tray_residual_g"})
    merged["observed_g"] = merged["target_fw_g"]
    merged["target_residual_g"] = merged["observed_g"] - merged["modelled_g"]
    return merged


def main() -> None:
    ML_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for trial in ("trial1", "trial2"):
        df = build_trial(trial)
        out = ML_DIR / f"checkpoint_residual_{trial}.csv"
        df.to_csv(out, index=False)
        frames.append(df)
        print(f"{trial}: {len(df)} rows -> {out.relative_to(PROJECT_ROOT)}")
        print(
            f"  residual_g range: {df['residual_g'].min():+.1f} .. "
            f"{df['residual_g'].max():+.1f} g"
        )

    combined = pd.concat(frames, ignore_index=True)
    combined_path = ML_DIR / "checkpoint_residual_combined.csv"
    combined.to_csv(combined_path, index=False)
    print(f"combined: {len(combined)} rows -> {combined_path.relative_to(PROJECT_ROOT)}")

    plant_out = ML_DIR / "plant" / "plant_residual_trial1.csv"
    plant_out.parent.mkdir(parents=True, exist_ok=True)
    plant_df = build_plant_trial1()
    plant_df.to_csv(plant_out, index=False)
    print(
        f"plant trial1: {len(plant_df)} rows -> {plant_out.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"  plant residual_g range: {plant_df['target_residual_g'].min():+.1f} .. "
        f"{plant_df['target_residual_g'].max():+.1f} g"
    )


if __name__ == "__main__":
    main()

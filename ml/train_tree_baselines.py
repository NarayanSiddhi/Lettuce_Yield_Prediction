"""
LOO evaluation for tree models on ml/checkpoint_trial1_image.csv.

  python3 ml/train_tree_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline

ML_DIR = Path(__file__).resolve().parent
TARGET = "target_median_fw_g"
DROP = {
    "sample_id",
    "trial",
    "day",
    "date",
    "window_start",
    "window_end",
    "label_source",
    TARGET,
}


def loo_eval(name: str, model, X: pd.DataFrame, y: pd.Series) -> None:
    preds = np.empty(len(y))
    loo = LeaveOneOut()
    for tr, te in loo.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        preds[te[0]] = model.predict(X.iloc[te])[0]
    print(
        f"  {name:22s}  MAE={mean_absolute_error(y, preds):.2f} g  "
        f"R²={r2_score(y, preds):.3f}"
    )


def main() -> None:
    path = ML_DIR / "checkpoint_trial1_image.csv"
    if not path.exists():
        print("Run: bash run_ml_data_pipeline.sh", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    y = pd.to_numeric(df[TARGET], errors="coerce")
    X = df.drop(columns=[c for c in DROP if c in df.columns], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")

    print(f"Loaded {path.name}: {len(df)} rows, {X.shape[1]} features\nLOO metrics:")

    rf = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42)),
        ]
    )
    loo_eval("RandomForest", rf, X, y)

    try:
        from xgboost import XGBRegressor

        xgb = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=80,
                        max_depth=3,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        random_state=42,
                    ),
                ),
            ]
        )
        loo_eval("XGBoost", xgb, X, y)
    except ImportError:
        print("  XGBoost not installed — pip install xgboost")


if __name__ == "__main__":
    main()

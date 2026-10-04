"""
Train and compare multiple yield prediction models on ML-ready checkpoint tables.

Default: leave-one-out evaluation on ml/checkpoint_trial1_image.csv.

Usage:
  python3 ml/train_yield_models.py
  python3 ml/train_yield_models.py --dataset checkpoint_trial1_compact.csv
  python3 ml/train_yield_models.py --trial both --variant compact
  python3 ml/train_yield_models.py --dataset checkpoint_combined.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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


@dataclass(frozen=True)
class Result:
    name: str
    n_features: int
    mae_g: float
    rmse_g: float
    r2: float


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 2:
        return float("nan")
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return 1.0 - (ss_res / ss_tot if ss_tot > 0 else float("inf"))


def metrics(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    mae = float(np.mean(np.abs(y - pred)))
    rmse = float(math.sqrt(float(np.mean((y - pred) ** 2))))
    return mae, rmse, _r2(y, pred)


def load_xy(dataset_name: str) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    # pandas is used only for CSV parsing; sklearn models run on numpy arrays
    import pandas as pd

    path = ML_DIR / dataset_name
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: bash run_ml_data_pipeline.sh")

    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise SystemExit(f"Missing target column {TARGET} in {path.name}")

    y = pd.to_numeric(df[TARGET], errors="coerce").to_numpy(dtype=float)
    Xdf = df.drop(columns=[c for c in DROP if c in df.columns], errors="ignore")
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
    feat_cols = list(Xdf.columns)
    X = Xdf.to_numpy(dtype=float)

    # keep a minimal row view for saving predictions
    keep_cols = [c for c in ["sample_id", "trial", "day", "date", "window_days_count"] if c in df.columns]
    row_meta = df[keep_cols].to_dict(orient="records") if keep_cols else [{} for _ in range(len(df))]
    return X, y, feat_cols, row_meta


def loo_predict(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(y)
    pred = np.empty(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        model.fit(X[mask], y[mask])
        pred[i] = float(model.predict(X[i : i + 1])[0])
    return pred


def build_models(random_state: int = 42):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    models: list[tuple[str, object]] = []

    # Linear baselines (strong regularization for n=8)
    models.append(
        (
            "Ridge",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", Ridge(alpha=10.0, random_state=random_state)),
                ]
            ),
        )
    )
    models.append(
        (
            "ElasticNet",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", ElasticNet(alpha=0.05, l1_ratio=0.25, random_state=random_state, max_iter=50_000)),
                ]
            ),
        )
    )

    # Nonlinear baseline
    models.append(
        (
            "SVR-RBF",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", SVR(C=10.0, gamma="scale", epsilon=5.0)),
                ]
            ),
        )
    )

    # Tree baselines
    models.append(
        (
            "RandomForest",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("model", RandomForestRegressor(n_estimators=400, max_depth=4, random_state=random_state)),
                ]
            ),
        )
    )
    models.append(
        (
            "GradBoost",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("model", GradientBoostingRegressor(random_state=random_state, n_estimators=300, max_depth=2)),
                ]
            ),
        )
    )

    # Optional: XGBoost
    try:
        from xgboost import XGBRegressor  # type: ignore

        models.append(
            (
                "XGBoost",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        (
                            "model",
                            XGBRegressor(
                                n_estimators=120,
                                max_depth=3,
                                learning_rate=0.08,
                                subsample=0.9,
                                colsample_bytree=0.8,
                                reg_lambda=2.0,
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
            )
        )
    except Exception:
        pass

    return models


def save_predictions(out_path: Path, meta_rows: list[dict], y: np.ndarray, preds_by_model: dict[str, np.ndarray]) -> None:
    fields = list(meta_rows[0].keys()) if meta_rows and meta_rows[0] else []
    fields += ["target_median_fw_g", "model", "pred_median_fw_g", "abs_error_g"]

    rows_out: list[dict] = []
    for i in range(len(y)):
        for model_name, pred in preds_by_model.items():
            r = dict(meta_rows[i]) if meta_rows else {}
            r["target_median_fw_g"] = float(y[i])
            r["model"] = model_name
            r["pred_median_fw_g"] = float(pred[i])
            r["abs_error_g"] = float(abs(pred[i] - y[i]))
            rows_out.append(r)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)

def run_one_dataset(dataset: str, seed: int) -> None:
    X, y, _feat_cols, meta = load_xy(dataset)
    print(f"\n=== {dataset} ===")
    print(f"Loaded {dataset}: {len(y)} rows, {X.shape[1]} features")

    # mean baseline
    mean_pred = np.full_like(y, float(np.mean(y)))
    mean_mae, mean_rmse, mean_r2 = metrics(y, mean_pred)
    print(f"LOO metrics (exploratory):")
    print(f"  Mean baseline          MAE={mean_mae:.2f} g  RMSE={mean_rmse:.2f} g  R²={mean_r2:.3f}")

    preds_by_model: dict[str, np.ndarray] = {}
    results: list[Result] = []
    for name, model in build_models(random_state=seed):
        pred = loo_predict(model, X, y)
        preds_by_model[name] = pred
        mae, rmse, r2v = metrics(y, pred)
        results.append(Result(name=name, n_features=X.shape[1], mae_g=mae, rmse_g=rmse, r2=r2v))
        print(f"  {name:20s}  MAE={mae:.2f} g  RMSE={rmse:.2f} g  R²={r2v:.3f}")

    results.sort(key=lambda r: r.mae_g)

    out_pred = ML_DIR / f"predictions_{Path(dataset).stem}_loo.csv"
    save_predictions(out_pred, meta, y, preds_by_model)
    print(f"Saved predictions: {out_pred.relative_to(ML_DIR.parent)}")

    out_sum = ML_DIR / f"results_{Path(dataset).stem}_loo.csv"
    with out_sum.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "n_features", "mae_g", "rmse_g", "r2"])
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "model": r.name,
                    "n_features": r.n_features,
                    "mae_g": f"{r.mae_g:.4f}",
                    "rmse_g": f"{r.rmse_g:.4f}",
                    "r2": f"{r.r2:.6f}",
                }
            )
    print(f"Saved summary: {out_sum.relative_to(ML_DIR.parent)}")


def resolve_datasets(dataset: str | None, trial: str, variant: str) -> list[str]:
    if dataset:
        return [dataset]
    if trial not in {"trial1", "trial2", "both"}:
        raise SystemExit("--trial must be one of: trial1, trial2, both")
    if variant not in {"image", "compact", "full"}:
        raise SystemExit("--variant must be one of: image, compact, full")

    suffix = {"image": "image", "compact": "compact", "full": ""}[variant]
    def name(t: str) -> str:
        if suffix:
            return f"checkpoint_{t}_{suffix}.csv"
        return f"checkpoint_{t}.csv"

    if trial == "both":
        return [name("trial1"), name("trial2")]
    return [name(trial)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="", help="CSV under ml/ (overrides --trial/--variant)")
    p.add_argument("--trial", default="trial1", choices=["trial1", "trial2", "both"], help="Which trial to run")
    p.add_argument("--variant", default="image", choices=["image", "compact", "full"], help="Feature set to use")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    datasets = resolve_datasets(args.dataset.strip() or None, args.trial, args.variant)
    for ds in datasets:
        run_one_dataset(ds, seed=args.seed)

    print("\nNote: only 8 labeled checkpoints per trial — treat metrics as exploratory, not final validation.")


if __name__ == "__main__":
    main()


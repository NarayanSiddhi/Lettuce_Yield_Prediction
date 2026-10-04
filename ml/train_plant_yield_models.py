"""
Train yield models on plant-level tables under ml/plant/.

Writes ONLY under ml/plant/. Does not modify existing checkpoint LOO result files.

Usage:
  python3 ml/train_plant_yield_models.py
  python3 ml/train_plant_yield_models.py --variant sensors
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PLANT_DIR = Path(__file__).resolve().parent / "plant"
TARGET = "target_fw_g"
DROP = {
    "sample_id",
    "trial",
    "date",
    "window_start",
    "window_end",
    "label_source",
    "tray_median_fw_g",
    TARGET,
}
# day and plant_id stay as features


@dataclass(frozen=True)
class Result:
    name: str
    feature_set: str
    split: str
    n_features: int
    n_rows: int
    mae_g: float
    rmse_g: float
    r2: float
    day_median_mae_g: float


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


def day_median_mae(days: np.ndarray, y: np.ndarray, pred: np.ndarray) -> float:
    """MAE of predicted-plant median vs true-plant median, one value per harvest day."""
    err = []
    for d in np.unique(days):
        mask = days == d
        err.append(abs(float(np.median(pred[mask])) - float(np.median(y[mask]))))
    return float(np.mean(err)) if err else float("nan")


def load_xy(dataset_name: str):
    path = PLANT_DIR / dataset_name
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: python3 export_plant_ml_datasets.py")

    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise SystemExit(f"Missing {TARGET} in {path.name}")

    y = pd.to_numeric(df[TARGET], errors="coerce").to_numpy(dtype=float)
    days = pd.to_numeric(df["day"], errors="coerce").to_numpy(dtype=int)
    plants = pd.to_numeric(df["plant_id"], errors="coerce").to_numpy(dtype=int)
    Xdf = df.drop(columns=[c for c in DROP if c in df.columns], errors="ignore")
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
    X = Xdf.to_numpy(dtype=float)
    keep_cols = [c for c in ["sample_id", "trial", "day", "plant_id", "date"] if c in df.columns]
    meta = df[keep_cols].to_dict(orient="records")
    return X, y, days, plants, list(Xdf.columns), meta


def loo_predict(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(y)
    pred = np.empty(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        model.fit(X[mask], y[mask])
        pred[i] = float(model.predict(X[i : i + 1])[0])
    return pred


def lodo_predict(model, X: np.ndarray, y: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Leave-one-harvest-day-out: train on all other days, test every plant on the held-out day."""
    pred = np.empty(len(y), dtype=float)
    for d in np.unique(days):
        te = days == d
        tr = ~te
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te]).astype(float)
    return pred


def lopo_predict(model, X: np.ndarray, y: np.ndarray, plants: np.ndarray) -> np.ndarray:
    """Leave-one-plant-out: train on other plants, test all days of held-out plant."""
    pred = np.empty(len(y), dtype=float)
    for p in np.unique(plants):
        te = plants == p
        tr = ~te
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te]).astype(float)
    return pred


def build_models(random_state: int = 42):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    models: list[tuple[str, object]] = [
        (
            "Ridge",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", Ridge(alpha=10.0, random_state=random_state)),
                ]
            ),
        ),
        (
            "ElasticNet",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", ElasticNet(alpha=0.05, l1_ratio=0.25, random_state=random_state, max_iter=50_000)),
                ]
            ),
        ),
        (
            "SVR-RBF",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", SVR(C=10.0, gamma="scale", epsilon=5.0)),
                ]
            ),
        ),
        (
            "RandomForest",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=400,
                            max_depth=4,
                            random_state=random_state,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "GradBoost",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("model", GradientBoostingRegressor(random_state=random_state, n_estimators=300, max_depth=2)),
                ]
            ),
        ),
    ]
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
                                n_jobs=1,
                            ),
                        ),
                    ]
                ),
            )
        )
    except Exception:
        pass
    return models


def save_predictions(
    out_path: Path,
    meta_rows: list[dict],
    y: np.ndarray,
    preds_by_model: dict[str, np.ndarray],
    split: str,
) -> None:
    fields = list(meta_rows[0].keys()) if meta_rows and meta_rows[0] else []
    fields += ["split", TARGET, "model", "pred_fw_g", "abs_error_g"]
    rows_out = []
    for i in range(len(y)):
        for model_name, pred in preds_by_model.items():
            r = dict(meta_rows[i]) if meta_rows else {}
            r["split"] = split
            r[TARGET] = float(y[i])
            r["model"] = model_name
            r["pred_fw_g"] = float(pred[i])
            r["abs_error_g"] = float(abs(pred[i] - y[i]))
            rows_out.append(r)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)


def save_results(path: Path, results: list[Result]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "feature_set",
                "model",
                "split",
                "n_features",
                "n_rows",
                "mae_g",
                "rmse_g",
                "r2",
                "day_median_mae_g",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "feature_set": r.feature_set,
                    "model": r.name,
                    "split": r.split,
                    "n_features": r.n_features,
                    "n_rows": r.n_rows,
                    "mae_g": f"{r.mae_g:.4f}",
                    "rmse_g": f"{r.rmse_g:.4f}",
                    "r2": f"{r.r2:.6f}",
                    "day_median_mae_g": f"{r.day_median_mae_g:.4f}",
                }
            )


def feature_set_name(dataset: str) -> str:
    stem = Path(dataset).stem
    if stem.endswith("_sensors"):
        return "sensors"
    if stem.endswith("_image"):
        return "image"
    if stem.endswith("_compact"):
        return "compact"
    if stem == "plant_checkpoint_trial1":
        return "full"
    return stem


def run_one_dataset(dataset: str, seed: int) -> list[Result]:
    X, y, days, plants, _feats, meta = load_xy(dataset)
    stem = Path(dataset).stem
    fset = feature_set_name(dataset)
    print(f"\n=== {dataset} (feature_set={fset}) ===")
    print(
        f"Loaded {len(y)} plant-days, {X.shape[1]} features, "
        f"{len(np.unique(days))} harvest days, {len(np.unique(plants))} plants"
    )

    all_results: list[Result] = []
    models = [("MeanBaseline", None)] + build_models(random_state=seed)

    for split_name, predict_fn in (
        ("loo", lambda model: loo_predict(model, X, y)),
        ("lodo", lambda model: lodo_predict(model, X, y, days)),
        ("lopo", lambda model: lopo_predict(model, X, y, plants)),
    ):
        print(f"\n  -- {split_name} --")
        preds_by_model: dict[str, np.ndarray] = {}
        split_results: list[Result] = []
        for name, model in models:
            if name == "MeanBaseline":
                pred = np.empty_like(y)
                if split_name == "loo":
                    for i in range(len(y)):
                        pred[i] = float(np.mean(y[np.arange(len(y)) != i]))
                elif split_name == "lodo":
                    for d in np.unique(days):
                        te = days == d
                        pred[te] = float(np.mean(y[~te]))
                else:  # lopo
                    for p in np.unique(plants):
                        te = plants == p
                        pred[te] = float(np.mean(y[~te]))
            else:
                pred = predict_fn(model)
            preds_by_model[name] = pred
            mae, rmse, r2v = metrics(y, pred)
            dmed = day_median_mae(days, y, pred)
            rec = Result(
                name=name,
                feature_set=fset,
                split=split_name,
                n_features=X.shape[1],
                n_rows=len(y),
                mae_g=mae,
                rmse_g=rmse,
                r2=r2v,
                day_median_mae_g=dmed,
            )
            split_results.append(rec)
            all_results.append(rec)
            print(
                f"    {name:16s}  plant MAE={mae:6.2f} g  R²={r2v:6.3f}  "
                f"day-median MAE={dmed:6.2f} g"
            )

        save_predictions(
            PLANT_DIR / f"predictions_{stem}_{split_name}.csv",
            meta,
            y,
            preds_by_model,
            split_name,
        )
        save_results(PLANT_DIR / f"results_{stem}_{split_name}.csv", split_results)

    return all_results


def write_ablation_summary(results: list[Result]) -> None:
    """Paper-friendly ablation table: best model per feature_set × split."""
    out = PLANT_DIR / "ablation_aim1_plant_fw.csv"
    save_results(out, results)
    # also best-per-cell excluding MeanBaseline
    best_rows = []
    keys = {(r.feature_set, r.split) for r in results}
    for fset, split in sorted(keys):
        cand = [r for r in results if r.feature_set == fset and r.split == split and r.name != "MeanBaseline"]
        if not cand:
            continue
        best = min(cand, key=lambda r: r.mae_g)
        best_rows.append(best)
    save_results(PLANT_DIR / "ablation_aim1_plant_fw_best.csv", best_rows)
    print(f"\nAblation tables:\n  {out}\n  {PLANT_DIR / 'ablation_aim1_plant_fw_best.csv'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant",
        default="all",
        choices=["all", "sensors", "image", "compact", "full"],
        help="Which plant-level feature set to train",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    names = {
        "sensors": "plant_checkpoint_trial1_sensors.csv",
        "image": "plant_checkpoint_trial1_image.csv",
        "compact": "plant_checkpoint_trial1_compact.csv",
        "full": "plant_checkpoint_trial1.csv",
    }
    datasets = list(names.values()) if args.variant == "all" else [names[args.variant]]

    combined: list[Result] = []
    for ds in datasets:
        combined.extend(run_one_dataset(ds, seed=args.seed))

    save_results(PLANT_DIR / "results_plant_all_variants.csv", combined)
    write_ablation_summary(combined)
    print(f"\nWrote all summaries under {PLANT_DIR}/")
    print("Existing ml/results_checkpoint_* files were not modified.")
    print(
        "Ablation grid: feature_sets={sensors,image,compact,full} × "
        "models={MeanBaseline,Ridge,ElasticNet,SVR-RBF,RandomForest,GradBoost,XGBoost} × "
        "splits={loo,lodo,lopo}"
    )


if __name__ == "__main__":
    main()

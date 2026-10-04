"""
Early-warning classifier: DAT ≤ 12 features → harvest shortfall label.

Label (professor definition):
  shortfall = harvest_fw_g <= 227 * (1 - pct_below)
  default pct_below = 0.20 → threshold ≈ 181.6 g

Trial 1 plant harvests never meet the strict 20% rule (lightest ≈ 185.8 g),
so this script also evaluates 10% and 15% thresholds for model comparison,
plus a regress-then-threshold path (predict final grams, then apply the cut).

Usage:
  python3 ml/train_early_warning.py
  python3 ml/train_early_warning.py --thresholds 0.10,0.15,0.20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

ML_DIR = Path(__file__).resolve().parent
PLANT_DIR = ML_DIR / "plant"
OUT_DIR = ML_DIR / "early_warning"
TARGET_G = 227.0
EARLY_DAYS = (0, 4, 8, 12)
HARVEST_DAY = 28

IMAGE_KEYS = [
    "img_mask_area_sum_wmean",
    "img_mask_area_mean_wmean",
    "img_coverage_frac_wmean",
    "img_plant_color_mean_wmean",
    "img_blur_mean_wmean",
    "win_img_mask_area_sum_slope",
    "win_img_mask_area_sum_delta",
    "win_img_coverage_frac_slope",
    "plant_img_mask_area_px_wmean",
    "plant_img_mask_area_px_last",
    "plant_img_plant_color_frac_wmean",
    "has_plant_cup_image",
]
SENSOR_KEYS = [
    "env_day_air_temp_c_mean_wmean",
    "env_night_air_temp_c_mean_wmean",
    "env_day_humidity_pct_mean_wmean",
    "env_night_humidity_pct_mean_wmean",
    "env_day_vpd_pa_mean_wmean",
    "env_night_vpd_pa_mean_wmean",
    "env_day_co2_ppm_mean_wmean",
    "env_act_day_light_duration_s_sum_wmean",
    "nutrient_ec_us_cm_mean_wmean",
    "nutrient_ph_mean_wmean",
    "exp_day_wmean",
]


@dataclass
class ClfResult:
    model: str
    variant: str
    threshold_pct: float
    threshold_g: float
    n_pos: int
    n_neg: int
    n_features: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    roc_auc: float
    mode: str
    harvest_mae_g: float = float("nan")


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def clf_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    if math.isnan(prec) or math.isnan(rec) or (prec + rec) == 0:
        f1 = float("nan")
    else:
        f1 = _safe_div(2 * prec * rec, prec + rec)
    tpr = _safe_div(tp, tp + fn)
    tnr = _safe_div(tn, tn + fp)
    bal = float(np.nanmean([tpr, tnr]))
    auc = float("nan")
    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auc = float("nan")
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "balanced_accuracy": bal,
        "roc_auc": auc,
    }


def variant_columns(variant: str, available: list[str]) -> list[str]:
    if variant == "image":
        keys = IMAGE_KEYS
    elif variant == "sensors":
        keys = SENSOR_KEYS
    else:
        keys = IMAGE_KEYS + SENSOR_KEYS
    return [c for c in keys if c in available]


def drop_bad_columns(X: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for c in X.columns:
        s = pd.to_numeric(X[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return X[keep]


def build_plant_early_table(variant: str) -> tuple[pd.DataFrame, list[str]]:
    path = PLANT_DIR / "plant_checkpoint_trial1_compact.csv"
    if variant == "image":
        path = PLANT_DIR / "plant_checkpoint_trial1_image.csv"
    elif variant == "sensors":
        path = PLANT_DIR / "plant_checkpoint_trial1_sensors.csv"

    df = pd.read_csv(path)
    residual = pd.read_csv(PLANT_DIR / "plant_residual_trial1.csv")
    res_keep = residual[["plant_id", "day", "modelled_g", "target_residual_g"]].copy()
    feat_cols = variant_columns(variant, list(df.columns))

    rows = []
    for plant_id, g in df.groupby("plant_id"):
        g = g.sort_values("day")
        harvest = g[g["day"] == HARVEST_DAY]
        if harvest.empty:
            continue
        harvest_fw = float(harvest.iloc[0]["target_fw_g"])

        early = g[g["day"].isin(EARLY_DAYS)].copy()
        if early.empty:
            continue

        fw_map = {int(r.day): float(r.target_fw_g) for r in early.itertuples()}
        feat: dict[str, float] = {"plant_id": int(plant_id), "harvest_fw_g": harvest_fw}
        for d in EARLY_DAYS:
            feat[f"fw_dat{d}"] = fw_map.get(d, np.nan)
        feat["fw_gain_0_to_12"] = feat["fw_dat12"] - feat["fw_dat0"]
        feat["fw_rel_gain_0_to_12"] = _safe_div(feat["fw_gain_0_to_12"], max(feat["fw_dat0"], 1e-6))

        snap = early[early["day"] == 12]
        if snap.empty:
            snap = early.iloc[[-1]]
        for c in feat_cols:
            val = pd.to_numeric(snap.iloc[0][c], errors="coerce") if c in snap.columns else np.nan
            feat[f"dat12_{c}"] = float(val) if pd.notna(val) else np.nan

        rg = res_keep[(res_keep["plant_id"] == plant_id) & (res_keep["day"] == 12)]
        if not rg.empty:
            feat["dat12_modelled_g"] = float(rg.iloc[0]["modelled_g"])
            feat["dat12_target_residual_g"] = float(rg.iloc[0]["target_residual_g"])
        else:
            feat["dat12_modelled_g"] = np.nan
            feat["dat12_target_residual_g"] = np.nan

        feat["dat12_gap_to_target_mid"] = feat["fw_dat12"] - 56.0
        rows.append(feat)

    out = pd.DataFrame(rows).sort_values("plant_id").reset_index(drop=True)
    feature_df = drop_bad_columns(out.drop(columns=["plant_id", "harvest_fw_g"]))
    feature_cols = list(feature_df.columns)
    out = pd.concat([out[["plant_id", "harvest_fw_g"]], feature_df], axis=1)
    return out, feature_cols


def build_classifiers(random_state: int = 42):
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    models: list[tuple[str, object]] = [
        ("DummyMostFrequent", DummyClassifier(strategy="most_frequent")),
        (
            "LogisticRegression",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=5000,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        (
            "SVM-RBF",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        SVC(
                            kernel="rbf",
                            C=1.0,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
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
                        RandomForestClassifier(
                            n_estimators=200,
                            max_depth=3,
                            class_weight="balanced",
                            random_state=random_state,
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
                    (
                        "model",
                        GradientBoostingClassifier(
                            n_estimators=100,
                            max_depth=2,
                            learning_rate=0.1,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
    ]
    try:
        from xgboost import XGBClassifier  # type: ignore

        models.append(
            (
                "XGBoost",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        (
                            "model",
                            XGBClassifier(
                                n_estimators=80,
                                max_depth=2,
                                learning_rate=0.1,
                                subsample=0.9,
                                colsample_bytree=0.8,
                                reg_lambda=2.0,
                                eval_metric="logloss",
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


def build_regressors(random_state: int = 42):
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
                    ("model", Ridge(alpha=10.0)),
                ]
            ),
        ),
        (
            "ElasticNet",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", ElasticNet(alpha=0.05, l1_ratio=0.25, max_iter=50_000, random_state=random_state)),
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
                    ("model", RandomForestRegressor(n_estimators=200, max_depth=3, random_state=random_state)),
                ]
            ),
        ),
        (
            "GradBoost",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("model", GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=random_state)),
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
                                n_estimators=80,
                                max_depth=2,
                                learning_rate=0.1,
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


def _predict_proba_positive(model, X_row: np.ndarray) -> float:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_row)
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            last = list(model.named_steps.values())[-1]
            classes = getattr(last, "classes_", None)
        if classes is not None and len(classes) == 1:
            return 1.0 if int(classes[0]) == 1 else 0.0
        if classes is not None and 1 in list(classes):
            idx = list(classes).index(1)
            return float(proba[0, idx])
        if proba.shape[1] == 2:
            return float(proba[0, 1])
        return float(proba[0, -1])
    if hasattr(model, "decision_function"):
        return float(model.decision_function(X_row)[0])
    return float(model.predict(X_row)[0])


def leave_one_plant_classify(model, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    pred = np.zeros(n, dtype=int)
    score = np.full(n, np.nan, dtype=float)
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        y_tr = y[tr]
        if len(np.unique(y_tr)) < 2:
            pred[i] = int(y_tr[0])
            score[i] = float(y_tr[0])
            continue
        try:
            model.fit(X[tr], y_tr)
        except Exception:
            pred[i] = int(np.bincount(y_tr.astype(int)).argmax())
            score[i] = float(pred[i])
            continue
        pred[i] = int(model.predict(X[i : i + 1])[0])
        try:
            score[i] = _predict_proba_positive(model, X[i : i + 1])
        except Exception:
            score[i] = float(pred[i])
    return pred, score


def leave_one_plant_regress(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(y)
    pred = np.empty(n, dtype=float)
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        model.fit(X[tr], y[tr])
        pred[i] = float(model.predict(X[i : i + 1])[0])
    return pred


def tray_shortfall_summary(threshold_g: float) -> list[dict]:
    rows = []
    for trial, harvest_obs in (("trial1", 219.8), ("trial2", 179.8)):
        fit = pd.read_csv(ML_DIR.parent / "Lettuce_model" / "Python" / "outputs" / f"{trial}_fit.csv")
        r12 = fit[fit["DAT"] == 12].iloc[0]
        shortfall = float(harvest_obs) <= threshold_g
        rows.append(
            {
                "trial": trial,
                "harvest_observed_g": harvest_obs,
                "shortfall_label": int(shortfall),
                "dat12_observed_g": float(r12["observed_g"]),
                "dat12_modelled_g": float(r12["modelled_g"]),
                "dat12_residual_g": float(r12["residual_g"]),
                "rule_warn_if_dat12_below_target_mid_56": int(float(r12["observed_g"]) < 56.0),
                "rule_warn_if_dat12_residual_neg": int(float(r12["residual_g"]) < 0.0),
                "threshold_g": threshold_g,
            }
        )
    return rows


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", default="0.10,0.15,0.20")
    parser.add_argument("--variants", default="compact,image,sensors")
    args = parser.parse_args()

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[ClfResult] = []
    pred_rows: list[dict] = []

    for variant in variants:
        table, feat_cols = build_plant_early_table(variant)
        table_path = OUT_DIR / f"plant_early_warning_trial1_{variant}.csv"
        table.to_csv(table_path, index=False)
        print(f"Wrote {table_path} ({len(table)} plants, {len(feat_cols)} features)", flush=True)

        X = table[feat_cols].to_numpy(dtype=float)
        harvest = table["harvest_fw_g"].to_numpy(dtype=float)
        plant_ids = table["plant_id"].to_numpy(dtype=int)

        reg_preds: dict[str, np.ndarray] = {}
        for name, model in build_regressors():
            reg_preds[name] = leave_one_plant_regress(model, X, harvest)
            mae = float(np.mean(np.abs(reg_preds[name] - harvest)))
            print(f"  [{variant}] regress {name}: harvest MAE={mae:.2f} g", flush=True)

        for pct in thresholds:
            thr_g = TARGET_G * (1.0 - pct)
            y = (harvest <= thr_g).astype(int)
            n_pos, n_neg = int(y.sum()), int((1 - y).sum())
            print(
                f"\n=== {variant} | {pct:.0%} below target (≤ {thr_g:.1f} g) | pos={n_pos} neg={n_neg} ===",
                flush=True,
            )

            for name, model in build_classifiers():
                if n_pos == 0:
                    all_results.append(
                        ClfResult(
                            model=name,
                            variant=variant,
                            threshold_pct=pct,
                            threshold_g=thr_g,
                            n_pos=n_pos,
                            n_neg=n_neg,
                            n_features=X.shape[1],
                            accuracy=float("nan"),
                            precision=float("nan"),
                            recall=float("nan"),
                            f1=float("nan"),
                            balanced_accuracy=float("nan"),
                            roc_auc=float("nan"),
                            mode="classifier",
                        )
                    )
                    continue

                y_pred, y_score = leave_one_plant_classify(model, X, y)
                m = clf_metrics(y, y_pred, y_score)
                all_results.append(
                    ClfResult(
                        model=name,
                        variant=variant,
                        threshold_pct=pct,
                        threshold_g=thr_g,
                        n_pos=n_pos,
                        n_neg=n_neg,
                        n_features=X.shape[1],
                        accuracy=m["accuracy"],
                        precision=m["precision"],
                        recall=m["recall"],
                        f1=m["f1"],
                        balanced_accuracy=m["balanced_accuracy"],
                        roc_auc=m["roc_auc"],
                        mode="classifier",
                    )
                )
                for i in range(len(y)):
                    pred_rows.append(
                        {
                            "mode": "classifier",
                            "variant": variant,
                            "model": name,
                            "threshold_pct": pct,
                            "threshold_g": thr_g,
                            "plant_id": int(plant_ids[i]),
                            "harvest_fw_g": float(harvest[i]),
                            "y_true": int(y[i]),
                            "y_pred": int(y_pred[i]),
                            "y_score": float(y_score[i]),
                        }
                    )
                print(
                    f"  clf {name:18s} acc={m['accuracy']:.2f} prec={m['precision']:.2f} "
                    f"rec={m['recall']:.2f} f1={m['f1']:.2f} bal={m['balanced_accuracy']:.2f} "
                    f"auc={m['roc_auc']:.2f}",
                    flush=True,
                )

            for name, y_hat in reg_preds.items():
                y_pred = (y_hat <= thr_g).astype(int)
                m = clf_metrics(y, y_pred, -y_hat)
                mae = float(np.mean(np.abs(y_hat - harvest)))
                all_results.append(
                    ClfResult(
                        model=name,
                        variant=variant,
                        threshold_pct=pct,
                        threshold_g=thr_g,
                        n_pos=n_pos,
                        n_neg=n_neg,
                        n_features=X.shape[1],
                        accuracy=m["accuracy"],
                        precision=m["precision"],
                        recall=m["recall"],
                        f1=m["f1"],
                        balanced_accuracy=m["balanced_accuracy"],
                        roc_auc=m["roc_auc"],
                        mode="regress_then_threshold",
                        harvest_mae_g=mae,
                    )
                )
                for i in range(len(y)):
                    pred_rows.append(
                        {
                            "mode": "regress_then_threshold",
                            "variant": variant,
                            "model": name,
                            "threshold_pct": pct,
                            "threshold_g": thr_g,
                            "plant_id": int(plant_ids[i]),
                            "harvest_fw_g": float(harvest[i]),
                            "pred_harvest_fw_g": float(y_hat[i]),
                            "y_true": int(y[i]),
                            "y_pred": int(y_pred[i]),
                            "y_score": float(-y_hat[i]),
                        }
                    )
                print(
                    f"  reg {name:18s} acc={m['accuracy']:.2f} prec={m['precision']:.2f} "
                    f"rec={m['recall']:.2f} f1={m['f1']:.2f} bal={m['balanced_accuracy']:.2f} "
                    f"auc={m['roc_auc']:.2f}",
                    flush=True,
                )

    results_path = OUT_DIR / "results_early_warning_comparison.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "variant",
                "model",
                "threshold_pct",
                "threshold_g",
                "n_pos",
                "n_neg",
                "n_features",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "balanced_accuracy",
                "roc_auc",
                "harvest_mae_g",
            ],
        )
        w.writeheader()
        for r in all_results:
            w.writerow(
                {
                    "mode": r.mode,
                    "variant": r.variant,
                    "model": r.model,
                    "threshold_pct": f"{r.threshold_pct:.2f}",
                    "threshold_g": f"{r.threshold_g:.2f}",
                    "n_pos": r.n_pos,
                    "n_neg": r.n_neg,
                    "n_features": r.n_features,
                    "accuracy": "" if math.isnan(r.accuracy) else f"{r.accuracy:.4f}",
                    "precision": "" if math.isnan(r.precision) else f"{r.precision:.4f}",
                    "recall": "" if math.isnan(r.recall) else f"{r.recall:.4f}",
                    "f1": "" if math.isnan(r.f1) else f"{r.f1:.4f}",
                    "balanced_accuracy": "" if math.isnan(r.balanced_accuracy) else f"{r.balanced_accuracy:.4f}",
                    "roc_auc": "" if math.isnan(r.roc_auc) else f"{r.roc_auc:.4f}",
                    "harvest_mae_g": "" if math.isnan(r.harvest_mae_g) else f"{r.harvest_mae_g:.4f}",
                }
            )
    print(f"\nWrote {results_path}", flush=True)

    pred_path = OUT_DIR / "predictions_early_warning_loo.csv"
    if pred_rows:
        keys = sorted({k for r in pred_rows for k in r.keys()})
        with pred_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(pred_rows)
        print(f"Wrote {pred_path}", flush=True)

    tray_rows = tray_shortfall_summary(TARGET_G * 0.8)
    tray_path = OUT_DIR / "tray_early_warning_rules_20pct.csv"
    pd.DataFrame(tray_rows).to_csv(tray_path, index=False)
    print(f"Wrote {tray_path}", flush=True)

    base = pd.read_csv(OUT_DIR / "plant_early_warning_trial1_compact.csv")
    summary = {
        "target_g": TARGET_G,
        "early_days": list(EARLY_DAYS),
        "harvest_day": HARVEST_DAY,
        "note": (
            "Trial 1 plant harvests: min≈185.8 g, so the official ≥20% below 227 g "
            "label has 0 positives. Compare models at 10% and 15% thresholds; "
            "also use regress-then-threshold for the 20% cut."
        ),
        "thresholds": {
            f"{p:.0%}": {
                "threshold_g": round(TARGET_G * (1 - p), 2),
                "n_pos_trial1_plants": int((base["harvest_fw_g"] <= TARGET_G * (1 - p)).sum()),
                "n_neg_trial1_plants": int((base["harvest_fw_g"] > TARGET_G * (1 - p)).sum()),
            }
            for p in thresholds
        },
    }
    (OUT_DIR / "early_warning_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_DIR / 'early_warning_summary.json'}", flush=True)


if __name__ == "__main__":
    run()

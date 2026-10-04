from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches
from docx.table import Table


PROJECT_ROOT = Path(__file__).resolve().parent
CROP_OUT = PROJECT_ROOT / "Lettuce_model" / "Python" / "outputs"
FIG_DIR = PROJECT_ROOT / "ml" / "figures"


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def set_table_borders(table: Table, size: str = "4", color: str = "000000") -> None:
    """Apply visible borders to every side of the table (and keep them on appends)."""
    tbl = table._tbl
    tblPr = tbl.tblPr
    if tblPr is None:
        tblPr = OxmlElement("w:tblPr")
        tbl.insert(0, tblPr)

    # Remove any existing borders definition so we don't stack duplicates
    for child in list(tblPr):
        if child.tag == qn("w:tblBorders"):
            tblPr.remove(child)

    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), size)  # eighths of a point; 4 ≈ 0.5 pt
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)
        borders.append(el)
    tblPr.append(borders)


def add_table(document: Document, title: str, columns: list[str], rows: list[dict], max_rows: int | None = None) -> None:
    document.add_paragraph(title)
    t = document.add_table(rows=1, cols=len(columns))
    hdr = t.rows[0].cells
    for i, c in enumerate(columns):
        hdr[i].text = c

    shown = rows if max_rows is None else rows[:max_rows]
    for r in shown:
        cells = t.add_row().cells
        for i, c in enumerate(columns):
            cells[i].text = str(r.get(c, ""))
    set_table_borders(t)
    document.add_paragraph("")


def add_figure(document: Document, path: Path, caption: str, width_in: float = 6.3) -> None:
    if not path.exists():
        document.add_paragraph(f"[Missing figure: {path.name}]")
        return
    document.add_picture(str(path), width=Inches(width_in))
    document.add_paragraph(caption)
    document.add_paragraph("")


def _best_setup_mae(rows: list[dict], setup: str) -> dict | None:
    hit = [r for r in rows if r.get("setup") == setup]
    if not hit:
        return None
    return min(hit, key=lambda r: float(r["mae_g"]))


def fmt(x: str, nd: int = 3) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    return f"{v:.{nd}f}"


def _best_non_baseline(rows: list[dict]) -> dict:
    usable = [r for r in rows if r.get("model") != "MeanBaseline"]
    return sorted(usable, key=lambda r: float(r["mae_g"]))[0]


def add_plant_level_section(doc: Document) -> None:
    """Section 6 — appended after the original tray-median results; does not edit sections 1–5."""
    plant_dir = PROJECT_ROOT / "ml" / "plant"

    doc.add_heading("6. Plant-Level Yield Prediction (Trial 1)", level=1)
    doc.add_paragraph(
        "Sections 3–5 report tray-median models (one row per harvest day). This section adds a follow-on "
        "experiment that uses each weighed plant as its own training example. Existing checkpoint tables and "
        "the results in Section 3 were not modified."
    )

    doc.add_heading("6.1 Setup", level=2)
    doc.add_paragraph(
        "The fresh-weight tracker already records New Fresh Weight (g) for Plant-IDs 1–10 on harvest days "
        "0, 4, 8, 12, 16, 20, 24, and 28. The tray-median pipeline collapsed those 10 weights to one number "
        "per day (8 rows). Here, each plant-day is kept as a separate row. The target is that plant’s own "
        "fresh weight (target_fw_g), not the tray median."
    )
    doc.add_paragraph(
        "This yields 79 labeled rows for Trial 1 (10 plants × 8 harvest days, except Day 8 is missing Plant 10, "
        "and a duplicate Plant-6 entry on Day 8 was averaged). Climate, nutrient, and actuator features are "
        "shared across plants on the same day (one tent). A plant-specific FastSAM cup image was attached when "
        "Plant-ID matched a camera cup_id in the same 4-day window (38 of 79 rows). Trial 2 is not included: "
        "it has no independent per-plant weights."
    )
    doc.add_paragraph("Four input sets were exported (new files only, under ml/plant/):")
    for item in [
        "Sensors: climate, nutrient, and actuator window features, plus plant ID and experimental day.",
        "Image: tray-level FastSAM canopy features plus plant-level cup image features when available.",
        "Compact: image features plus window-average sensors (same recipe as Section 3, expanded to plants).",
        "Full: all numeric window statistics plus plant-level image columns.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("6.2 Validation", level=2)
    doc.add_paragraph(
        "Two splits were run. Leave-one-out (LOO) hides one plant on one day but still shows the other plants "
        "from that same harvest day and the same plant on neighboring days, so it can look unrealistically good. "
        "Leave-one-harvest-day-out (LODO) hides all plants from one weighing day and trains on the other seven "
        "days. LODO is the fairer test of predicting a harvest that has not yet been seen, and is the result "
        "that should be used for comparison with Section 3."
    )
    doc.add_paragraph(
        "A mean baseline (predict the average weight of the training plants) has LODO MAE ≈ 67 g. "
        "Model MAE should be read against that number."
    )

    def add_plant_variant(title: str, fname: str) -> None:
        rows = read_csv_rows(plant_dir / fname)
        rows = [r for r in rows if r.get("model") != "MeanBaseline"]
        rows_sorted = sorted(rows, key=lambda r: float(r["mae_g"]))
        best = rows_sorted[0]
        doc.add_heading(title, level=3)
        doc.add_paragraph(
            f"Best model: {best['model']} "
            f"(plant MAE={float(best['mae_g']):.2f} g, R²={float(best['r2']):.3f}, "
            f"day-median MAE={float(best['day_median_mae_g']):.2f} g)."
        )
        table_rows = []
        for r in rows_sorted:
            table_rows.append({
                "model": r["model"],
                "n_features": r.get("n_features", ""),
                "mae_g": fmt(r.get("mae_g", ""), 2),
                "rmse_g": fmt(r.get("rmse_g", ""), 2),
                "r2": fmt(r.get("r2", ""), 3),
                "day_median_mae_g": fmt(r.get("day_median_mae_g", ""), 2),
            })
        add_table(
            doc,
            title=f"Leave-one-harvest-day-out results ({title.lower()}):",
            columns=["model", "n_features", "mae_g", "rmse_g", "r2", "day_median_mae_g"],
            rows=table_rows,
        )

    doc.add_heading("6.3 Leave-One-Harvest-Day-Out Results (Fair Comparison)", level=2)
    add_plant_variant("Image-only features", "results_plant_checkpoint_trial1_image_lodo.csv")
    add_plant_variant("Sensors-only features", "results_plant_checkpoint_trial1_sensors_lodo.csv")
    add_plant_variant("Compact (image + sensors)", "results_plant_checkpoint_trial1_compact_lodo.csv")
    add_plant_variant("Full feature set", "results_plant_checkpoint_trial1_lodo.csv")

    doc.add_heading("6.4 Comparison with Tray-Median Models", level=2)
    doc.add_paragraph(
        "On LODO, compact / sensors / full are essentially tied (best plant MAE 23.2–23.6 g, R² ≈ 0.75). "
        "Image-only is weaker (28.6 g), consistent with missing cup matches on many harvest windows. "
        "The extra columns in the full set do not improve accuracy over compact. "
        "Relative to Section 3.1 tray-median LOO (compact MAE 36.7 g; full MAE 31.7 g), the plant-level LODO "
        "error is lower, mainly because 79 plant-days provide more labeled rows than 8 tray medians."
    )

    loo_summary = []
    for label, fname in [
        ("Image-only", "results_plant_checkpoint_trial1_image_loo.csv"),
        ("Sensors-only", "results_plant_checkpoint_trial1_sensors_loo.csv"),
        ("Compact", "results_plant_checkpoint_trial1_compact_loo.csv"),
        ("Full", "results_plant_checkpoint_trial1_loo.csv"),
    ]:
        best = _best_non_baseline(read_csv_rows(plant_dir / fname))
        loo_summary.append({
            "input": label,
            "model": best["model"],
            "mae_g": fmt(best["mae_g"], 2),
            "r2": fmt(best["r2"], 3),
        })
    add_table(
        doc,
        title="Plant-level leave-one-out (optimistic; not used for model selection):",
        columns=["input", "model", "mae_g", "r2"],
        rows=loo_summary,
    )
    doc.add_paragraph(
        "Those LOO figures (MAE ≈ 3–5 g) should not be treated as the headline result: the model can still see "
        "other plants from the same day and the same plant on adjacent harvests, so it is largely interpolating "
        "the growth curve."
    )

    doc.add_heading("6.5 Interpretation and Selected Configuration", level=2)
    doc.add_paragraph(
        "For the next phase we keep the compact (image + sensors) set as the primary plant-level configuration, "
        "with XGBoost and leave-one-harvest-day-out as the reported metric. Compact matches the recommendation "
        "in Section 4, remains as accurate as the full dump, and still includes both cameras and environment. "
        "Sensors-only should be retained as an ablation: it is nearly identical to compact on LODO, which "
        "suggests that much of the plant-level signal is environment plus plant age rather than the extra "
        "photo columns. Image-only is kept only as a vision baseline."
    )
    for item in [
        "Plant-ID is used as a feature, so these models describe the 10 weighed plants rather than unseen cups.",
        "Cup image attachment assumes Plant-ID equals camera cup_id; that mapping has not been verified on the bench.",
        "All 10 plants share the same tent climate on a given day, so extra rows help image-to-weight modeling more than they create independent environment experiments.",
        "The target is still same-window (current harvest) weight, not a forecast of the next checkpoint.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    add_best_model_summary_section(doc)
    add_crop_model_hybrid_section(doc)
    add_early_warning_section(doc)


def add_crop_model_hybrid_section(doc: Document) -> None:
    """Section 7 — mechanistic crop model + residual hybrid ML comparisons."""
    ml_dir = PROJECT_ROOT / "ml"
    plant_dir = ml_dir / "plant"
    summary_path = CROP_OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    doc.add_heading("7. Mechanistic Crop Model and Residual Hybrid ML", level=1)
    doc.add_paragraph(
        "Sections 3 and 6 train ML models to predict fresh weight directly from images and sensors. "
        "This section adds a different baseline: a physics-based lettuce crop model (Van Henten growth "
        "structure with Farquhar–von Caemmerer–Berry photosynthesis) that predicts expected tray-median "
        "weight from light, temperature, CO₂, and humidity only. Nutrients/EC are deliberately left out "
        "of the crop model so that the residual (observed − modelled) is a clean target for image and "
        "EC analysis. We then compare three approaches on the same harvest checkpoints: crop model alone, "
        "ML predicting raw weight, and hybrid ML that predicts the residual and adds it back to the crop "
        "prediction."
    )

    doc.add_heading("7.1 Crop Model Setup", level=2)
    doc.add_paragraph(
        "Daily driver tables were built from Mycodo exports in Data collection/ (with CO₂ QC that drops "
        "implausible readings). PPFD was reconstructed from the documented light schedule because no PAR "
        "sensor was logged. One free parameter (canopy_efficiency) was fitted per trial by least squares "
        "in log fresh weight against the 4-day weigh-ins; all other physiology coefficients stay at "
        "literature or measured values."
    )
    for item in [
        "Trial 1: adaptive EC steering; PPFD step 200 → 445 µmol m⁻² s⁻¹ at DAT 12.",
        "Trial 2: fixed EC 1.4–1.6 mS/cm; constant ~445 µmol m⁻² s⁻¹.",
        "Outputs: Lettuce_model/Python/outputs/ (simulation CSVs, fit tables, growth_trajectories.png, summary.json).",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("7.2 Crop Model Fit Metrics", level=2)
    crop_rows = []
    for key, trial_label in (("trial1", "Trial 1"), ("trial2", "Trial 2")):
        block = summary.get(key, {})
        cal = block.get("calibration", {})
        met = block.get("metrics", {})
        if not met:
            continue
        crop_rows.append({
            "trial": trial_label,
            "canopy_efficiency": fmt(cal.get("canopy_efficiency", ""), 3),
            "MAE_g": fmt(met.get("MAE_g", ""), 2),
            "RMSE_g": fmt(met.get("RMSE_g", ""), 2),
            "MAPE_pct": fmt(met.get("MAPE_pct", ""), 1),
            "R2": fmt(met.get("R2", ""), 3),
            "bias_g": fmt(met.get("bias_g", ""), 2),
            "harvest_obs_g": fmt(met.get("final_observed_g", ""), 1),
            "harvest_model_g": fmt(met.get("final_modelled_g", ""), 1),
        })
    add_table(
        doc,
        title="Crop-model calibration and checkpoint metrics (DAT 4–28; DAT 0 excluded from fit):",
        columns=[
            "trial",
            "canopy_efficiency",
            "MAE_g",
            "RMSE_g",
            "MAPE_pct",
            "R2",
            "bias_g",
            "harvest_obs_g",
            "harvest_model_g",
        ],
        rows=crop_rows,
    )
    doc.add_paragraph(
        "Observed harvest difference (Trial 1 − Trial 2) is +40.0 g; the crop model difference is +50.5 g. "
        "So the light schedule alone is enough for the physics model to explain the full cross-trial harvest "
        "gap before EC is invoked. Both trials show the same residual shape: model ahead mid-cycle "
        "(DAT 16–20) and behind at harvest — a repeatable miss that residual ML is asked to explain."
    )

    doc.add_heading("7.3 Checkpoint Residuals (Observed − Modelled)", level=2)
    for trial_num, label in ((1, "Trial 1"), (2, "Trial 2")):
        fit_path = CROP_OUT / f"trial{trial_num}_fit.csv"
        if not fit_path.exists():
            continue
        fit_rows = read_csv_rows(fit_path)
        for r in fit_rows:
            for k in ("observed_g", "modelled_g", "residual_g", "residual_pct", "target_mid_g"):
                if k in r:
                    r[k] = fmt(r[k], 1)
        add_table(
            doc,
            title=f"{label} crop-model fit at harvest checkpoints:",
            columns=["DAT", "observed_g", "modelled_g", "residual_g", "residual_pct", "target_mid_g"],
            rows=fit_rows,
        )

    doc.add_heading("7.4 Crop Model Trajectory Figures", level=2)
    add_figure(
        doc,
        CROP_OUT / "growth_trajectories.png",
        "Figure 7.1. Crop-model growth trajectories vs observed medians and steering band "
        "(Lettuce_model/Python/outputs/growth_trajectories.png).",
        width_in=6.4,
    )
    add_figure(
        doc,
        CROP_OUT / "carbon_balance.png",
        "Figure 7.2. Crop-model carbon balance and related drivers "
        "(Lettuce_model/Python/outputs/carbon_balance.png).",
        width_in=6.4,
    )

    doc.add_heading("7.5 Tray-Level ML Results (8 Checkpoints, LODO)", level=2)
    doc.add_paragraph(
        "Crop-model metrics for these same checkpoints are reported in Section 7.2 "
        f"(Trial 1 MAE {fmt(summary.get('trial1', {}).get('metrics', {}).get('MAE_g', ''), 2)} g; "
        f"Trial 2 MAE {fmt(summary.get('trial2', {}).get('metrics', {}).get('MAE_g', ''), 2)} g). "
        "The tables below list ML results only. Visual side-by-side trajectories and error plots are in Section 7.8."
    )
    doc.add_paragraph(
        "Two ML setups were trained with leave-one-harvest-day-out (LODO): "
        "(1) raw weight — ML predicts observed grams from image/sensor window features; "
        "(2) hybrid residual — ML predicts residual_g, then final weight = modelled_g + predicted residual. "
        "Feature variants: image-only, compact (image + sensors), and sensors-only. "
        "Each row is the best model (lowest MAE) for that trial × feature set."
    )

    ml_raw_rows = []
    hybrid_rows = []
    for trial in ("trial1", "trial2"):
        for variant in ("image", "compact", "sensors"):
            path = ml_dir / f"results_residual_{trial}_{variant}_lodo.csv"
            if not path.exists():
                if variant == "image":
                    path = ml_dir / f"results_residual_{trial}_lodo.csv"
                else:
                    continue
            if not path.exists():
                continue
            rows = read_csv_rows(path)
            hybrid = (
                _best_setup_mae(rows, f"{variant}_hybrid")
                or _best_setup_mae(rows, "hybrid_residual")
            )
            raw = (
                _best_setup_mae(rows, f"{variant}_raw")
                or _best_setup_mae(rows, "image_raw")
            )
            if raw:
                ml_raw_rows.append({
                    "trial": trial,
                    "feature_set": variant,
                    "best_model": raw.get("model", ""),
                    "n_features": raw.get("n_features", ""),
                    "mae_g": fmt(raw["mae_g"], 2),
                    "rmse_g": fmt(raw.get("rmse_g", ""), 2),
                    "r2": fmt(raw.get("r2", ""), 3),
                })
            if hybrid:
                hybrid_rows.append({
                    "trial": trial,
                    "feature_set": variant,
                    "best_model": hybrid.get("model", ""),
                    "n_features": hybrid.get("n_features", ""),
                    "mae_g": fmt(hybrid["mae_g"], 2),
                    "rmse_g": fmt(hybrid.get("rmse_g", ""), 2),
                    "r2": fmt(hybrid.get("r2", ""), 3),
                })

    add_table(
        doc,
        title="ML predicting raw fresh weight (LODO; best model per feature set):",
        columns=["trial", "feature_set", "best_model", "n_features", "mae_g", "rmse_g", "r2"],
        rows=ml_raw_rows,
    )
    add_table(
        doc,
        title="Hybrid residual ML (LODO; best model per feature set):",
        columns=["trial", "feature_set", "best_model", "n_features", "mae_g", "rmse_g", "r2"],
        rows=hybrid_rows,
    )

    doc.add_heading("7.6 Hold-Out Results (Train Trial 1 → Test Trial 2)", level=2)
    holdout_path = ml_dir / "results_residual_holdout_trial2_compact.csv"
    if holdout_path.exists():
        hold_rows = read_csv_rows(holdout_path)
        doc.add_paragraph(
            "Models were trained on all Trial 1 checkpoints and evaluated on Trial 2 using compact "
            "(image + sensor) features. Crop-model Trial 2 MAE is again the Section 7.2 value "
            f"({fmt(summary.get('trial2', {}).get('metrics', {}).get('MAE_g', ''), 2)} g). "
            "ML results for this hold-out are listed below; see Figure 7.7 for trajectories."
        )

        hold_raw = []
        hold_hybrid = []
        for r in hold_rows:
            setup = r.get("setup", "")
            row = {
                "model": r.get("model", ""),
                "n_features": r.get("n_features", ""),
                "mae_g": fmt(r.get("mae_g", ""), 2),
                "rmse_g": fmt(r.get("rmse_g", ""), 2),
                "r2": fmt(r.get("r2", ""), 3),
            }
            if "raw" in setup and "crop" not in setup:
                hold_raw.append(row)
            elif "hybrid" in setup:
                hold_hybrid.append(row)

        hold_raw = sorted(hold_raw, key=lambda x: float(x["mae_g"]))
        hold_hybrid = sorted(hold_hybrid, key=lambda x: float(x["mae_g"]))

        if hold_raw:
            add_table(
                doc,
                title="Hold-out Trial 2 — ML predicting raw fresh weight (compact features):",
                columns=["model", "n_features", "mae_g", "rmse_g", "r2"],
                rows=hold_raw,
            )
        if hold_hybrid:
            add_table(
                doc,
                title="Hold-out Trial 2 — hybrid residual ML (compact features):",
                columns=["model", "n_features", "mae_g", "rmse_g", "r2"],
                rows=hold_hybrid,
            )

    doc.add_heading("7.7 Plant-Level ML Results (79 Rows, Trial 1 LODO)", level=2)
    doc.add_paragraph(
        "Each row is one weighed plant on one harvest day (target_fw_g). "
        "The crop prior used inside the hybrid setup is the tray-level modelled_g from Section 7.2 "
        f"(crop-model Trial 1 MAE {fmt(summary.get('trial1', {}).get('metrics', {}).get('MAE_g', ''), 2)} g on tray medians). "
        "Tables below report ML metrics only."
    )
    plant_raw = []
    plant_hybrid = []
    for variant in ("image", "compact", "sensors"):
        path = plant_dir / f"results_residual_plant_{variant}_lodo.csv"
        if not path.exists():
            continue
        rows = read_csv_rows(path)
        hybrid = _best_setup_mae(rows, f"{variant}_hybrid")
        raw = _best_setup_mae(rows, f"{variant}_raw")
        if raw:
            plant_raw.append({
                "feature_set": variant,
                "best_model": raw.get("model", ""),
                "n_features": raw.get("n_features", ""),
                "mae_g": fmt(raw["mae_g"], 2),
                "rmse_g": fmt(raw.get("rmse_g", ""), 2),
                "r2": fmt(raw.get("r2", ""), 3),
            })
        if hybrid:
            plant_hybrid.append({
                "feature_set": variant,
                "best_model": hybrid.get("model", ""),
                "n_features": hybrid.get("n_features", ""),
                "mae_g": fmt(hybrid["mae_g"], 2),
                "rmse_g": fmt(hybrid.get("rmse_g", ""), 2),
                "r2": fmt(hybrid.get("r2", ""), 3),
            })
    if plant_raw:
        add_table(
            doc,
            title="Plant-level LODO — ML predicting raw plant fresh weight:",
            columns=["feature_set", "best_model", "n_features", "mae_g", "rmse_g", "r2"],
            rows=plant_raw,
        )
    if plant_hybrid:
        add_table(
            doc,
            title="Plant-level LODO — hybrid residual ML:",
            columns=["feature_set", "best_model", "n_features", "mae_g", "rmse_g", "r2"],
            rows=plant_hybrid,
        )
        doc.add_paragraph(
            "Plant leave-one-out (LOO) MAE values of roughly 4–5 g appear in earlier plant-level runs "
            "but are optimistic (same-day plants remain visible) and are not used as the headline metric."
        )

    doc.add_heading("7.8 Figures (Crop Model and ML Together)", level=2)
    doc.add_paragraph(
        "The following figures show crop-model and ML predictions on the same axes for visual comparison. "
        "Numeric results remain in the separate tables above."
    )
    fig_specs = [
        (
            FIG_DIR / "growth_comparison_lodo.png",
            "Figure 7.3. Growth trajectories: observed vs crop model vs hybrid residual ML vs image-only ML "
            "(leave-one-harvest-day-out).",
        ),
        (
            FIG_DIR / "residual_comparison_lodo.png",
            "Figure 7.4. Prediction residuals by harvest day for crop, hybrid, and ML-only models.",
        ),
        (
            FIG_DIR / "scatter_observed_vs_pred.png",
            "Figure 7.5. Observed vs predicted scatter for crop, hybrid, and ML-only (both trials, LODO).",
        ),
        (
            FIG_DIR / "mae_summary_bars.png",
            "Figure 7.6. Best LODO MAE by feature set (image / compact / sensors).",
        ),
        (
            FIG_DIR / "holdout_trial2_compact.png",
            "Figure 7.7. Hold-out: train Trial 1 → predict Trial 2 (compact features).",
        ),
        (
            FIG_DIR / "mae_plant_lodo_bars.png",
            "Figure 7.8. Plant-level LODO MAE by feature set (79 plant-days).",
        ),
    ]
    for path, caption in fig_specs:
        add_figure(doc, path, caption, width_in=6.3)

    doc.add_heading("7.9 Interpretation and Next Steps", level=2)
    for item in [
        "Crop model (Section 7.2): tray-level MAE ~8–10 g, R² ≈ 0.95–0.98, without images or EC.",
        "ML raw weight models: higher LODO MAE on 8 tray checkpoints; see tables in 7.5.",
        "Hybrid residual ML: LODO and hold-out results in 7.5–7.6; trajectory overlays in 7.8.",
        "Early-warning classification (Aim 3) is reported in Section 8.",
        "Known crop-model limitations: assumed bench area, dry-matter fraction 4.5%, unlogged PPFD, and only two sequential trials (not true replicates).",
    ]:
        doc.add_paragraph(item, style="List Bullet")


def add_early_warning_section(doc: Document) -> None:
    """Section 8 — DAT ≤ 12 early-warning shortfall classifier."""
    ew_dir = PROJECT_ROOT / "ml" / "early_warning"
    results_path = ew_dir / "results_early_warning_comparison.csv"
    summary_path = ew_dir / "early_warning_summary.json"
    tray_path = ew_dir / "tray_early_warning_rules_20pct.csv"

    doc.add_heading("8. Early-Warning Classification (DAT ≤ 12)", level=1)
    doc.add_paragraph(
        "This section addresses Aim 3 from the project direction: at DAT 12, can the system identify "
        "plants that will finish well below the 227 g target? Features use only information available "
        "by DAT ≤ 12 (early fresh weights at DAT 0/4/8/12, DAT-12 crop-model residual, and gap to the "
        "steering mid-line). The harvest-day label is not used as an input."
    )

    doc.add_heading("8.1 Label Definition", level=2)
    doc.add_paragraph(
        "Shortfall label: harvest fresh weight ≤ 227 × (1 − pct_below). "
        "The professor definition uses pct_below = 20% (threshold ≈ 181.6 g). "
        "Trial 1 plant harvests never meet that cut (lightest plant ≈ 185.8 g, about 18% below target), "
        "so classifiers are also compared at 10% (≤ 204.3 g) and 15% (≤ 193.0 g), where positive "
        "examples exist."
    )

    label_rows = [
        {
            "threshold": "10% below",
            "cut_g": "204.3",
            "n_shortfall": "2",
            "n_ok": "8",
            "shortfall_plants": "5, 6",
        },
        {
            "threshold": "15% below",
            "cut_g": "193.0",
            "n_shortfall": "1",
            "n_ok": "9",
            "shortfall_plants": "5",
        },
        {
            "threshold": "20% below (official)",
            "cut_g": "181.6",
            "n_shortfall": "0",
            "n_ok": "10",
            "shortfall_plants": "—",
        },
    ]
    add_table(
        doc,
        title="Trial 1 plant-level shortfall counts (n = 10 plants):",
        columns=["threshold", "cut_g", "n_shortfall", "n_ok", "shortfall_plants"],
        rows=label_rows,
    )

    doc.add_heading("8.2 Methods", level=2)
    doc.add_paragraph(
        "Validation is leave-one-plant-out. Two modes are compared: (1) direct classifiers "
        "(Dummy most-frequent, Logistic Regression, SVM-RBF, Random Forest, Gradient Boosting, XGBoost); "
        "(2) regress-then-threshold — predict DAT-28 grams from DAT ≤ 12 features, then apply the "
        "shortfall cut. Feature variants: compact, image, and sensors. Within a single trial, tent "
        "climate at DAT 12 is identical across plants, so plant discrimination is driven mainly by "
        "early weights and the crop-model residual."
    )

    if not results_path.exists():
        doc.add_paragraph(f"[Missing early-warning results: {results_path}]")
        return

    rows = read_csv_rows(results_path)

    def _num(r: dict, key: str) -> float:
        try:
            return float(r.get(key) or "nan")
        except Exception:
            return float("nan")

    def _fmt_metric(r: dict, key: str, nd: int = 2) -> str:
        v = _num(r, key)
        if v != v:  # NaN
            return "—"
        return f"{v:.{nd}f}"

    # Compact @ 10%: classifiers
    clf_10 = [
        r
        for r in rows
        if r.get("mode") == "classifier"
        and r.get("variant") == "compact"
        and abs(_num(r, "threshold_pct") - 0.10) < 1e-9
    ]
    clf_10_sorted = sorted(
        clf_10,
        key=lambda r: (
            -(_num(r, "f1") if _num(r, "f1") == _num(r, "f1") else -1.0),
            -_num(r, "balanced_accuracy"),
            -_num(r, "roc_auc"),
        ),
    )
    clf_table = []
    for r in clf_10_sorted:
        clf_table.append(
            {
                "model": r["model"],
                "accuracy": _fmt_metric(r, "accuracy"),
                "precision": _fmt_metric(r, "precision"),
                "recall": _fmt_metric(r, "recall"),
                "f1": _fmt_metric(r, "f1"),
                "balanced_acc": _fmt_metric(r, "balanced_accuracy"),
                "roc_auc": _fmt_metric(r, "roc_auc"),
            }
        )

    doc.add_heading("8.3 Classifier Comparison (10% Below Target)", level=2)
    doc.add_paragraph(
        "Primary comparable setting: 10% below 227 g (≤ 204.3 g; plants 5 and 6 labeled shortfall). "
        "Compact feature set; leave-one-plant-out."
    )
    add_table(
        doc,
        title="Direct classifiers @ 10% shortfall threshold (compact features):",
        columns=["model", "accuracy", "precision", "recall", "f1", "balanced_acc", "roc_auc"],
        rows=clf_table,
    )

    reg_10 = [
        r
        for r in rows
        if r.get("mode") == "regress_then_threshold"
        and r.get("variant") == "compact"
        and abs(_num(r, "threshold_pct") - 0.10) < 1e-9
    ]
    reg_10_sorted = sorted(
        reg_10,
        key=lambda r: (
            -(_num(r, "f1") if _num(r, "f1") == _num(r, "f1") else -1.0),
            -_num(r, "roc_auc"),
        ),
    )
    reg_table = []
    for r in reg_10_sorted:
        reg_table.append(
            {
                "model": r["model"],
                "accuracy": _fmt_metric(r, "accuracy"),
                "precision": _fmt_metric(r, "precision"),
                "recall": _fmt_metric(r, "recall"),
                "f1": _fmt_metric(r, "f1"),
                "roc_auc": _fmt_metric(r, "roc_auc"),
                "harvest_mae_g": _fmt_metric(r, "harvest_mae_g"),
            }
        )

    doc.add_heading("8.4 Regress-then-Threshold (10% Below Target)", level=2)
    doc.add_paragraph(
        "Same label and split, but the model first predicts harvest grams from DAT ≤ 12 features, "
        "then warns if the prediction is ≤ 204.3 g. Harvest MAE is the leave-one-plant-out error "
        "of that gram forecast."
    )
    add_table(
        doc,
        title="Regress-then-threshold @ 10% shortfall (compact features):",
        columns=["model", "accuracy", "precision", "recall", "f1", "roc_auc", "harvest_mae_g"],
        rows=reg_table,
    )

    doc.add_heading("8.5 Official 20% Threshold and Tray Check", level=2)
    doc.add_paragraph(
        "At the official ≥20% below-target rule, Trial 1 has zero shortfall plants, so plant-level "
        "precision/recall are undefined and “accuracy = 1.0” only means every model predicted "
        "no warning. Trial 2’s tray median (179.8 g) does meet the 20% shortfall label. Simple "
        "DAT-12 tray rules:"
    )
    if tray_path.exists():
        tray_rows = read_csv_rows(tray_path)
        for r in tray_rows:
            r["threshold_g"] = fmt(r.get("threshold_g", ""), 1)
            r["harvest_observed_g"] = fmt(r.get("harvest_observed_g", ""), 1)
            r["dat12_observed_g"] = fmt(r.get("dat12_observed_g", ""), 1)
            r["dat12_modelled_g"] = fmt(r.get("dat12_modelled_g", ""), 1)
            r["dat12_residual_g"] = fmt(r.get("dat12_residual_g", ""), 1)
        add_table(
            doc,
            title="Tray-level descriptive rules at DAT 12 (20% shortfall label):",
            columns=[
                "trial",
                "harvest_observed_g",
                "shortfall_20pct_label",
                "dat12_observed_g",
                "dat12_modelled_g",
                "dat12_residual_g",
                "rule_warn_if_dat12_below_target_mid_56",
                "rule_warn_if_dat12_residual_neg",
            ],
            rows=tray_rows,
        )
        doc.add_paragraph(
            "Both trials are below the DAT-12 steering mid (56 g), so that rule alone false-alarms "
            "on Trial 1. A negative DAT-12 crop-model residual flags Trial 2 only, matching the "
            "true 20% harvest shortfall."
        )

    doc.add_heading("8.6 Interpretation", level=2)
    best_note = ""
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        bc = summary.get("best_at_10pct_classifier") or {}
        br = summary.get("best_at_10pct_regress_then_threshold") or {}
        if bc:
            best_note = (
                f"Best direct classifier at 10%: {bc.get('model')} "
                f"(F1={bc.get('f1')}, precision={bc.get('precision')}, recall={bc.get('recall')}). "
            )
        if br:
            best_note += (
                f"Best regress-then-threshold: {br.get('model')} "
                f"(F1={br.get('f1')}, ROC-AUC={br.get('roc_auc')}, "
                f"harvest MAE={br.get('harvest_mae_g')} g)."
            )
    for item in [
        best_note or "See tables above for model comparison at the 10% threshold.",
        "At 10%, top models catch 1 of 2 light plants with no false alarms (precision 1.0, recall 0.5, F1 ≈ 0.67).",
        "At 15% (only plant 5 positive), classifiers generally fail to recover that single case under leave-one-plant-out.",
        "The official 20% Aim-3 definition needs more shortfall examples (e.g. Trial 2 plant weights, or a third trial) before it can be scored as a plant-level classifier.",
        "Outputs: ml/early_warning/ (results_early_warning_comparison.csv, predictions_early_warning_loo.csv).",
    ]:
        if item:
            doc.add_paragraph(item, style="List Bullet")


def add_growth_forecast_section(doc: Document) -> None:
    """Section 9 — future growth forecasting (weight, canopy, future images)."""
    ml_dir = PROJECT_ROOT / "ml"
    wt_dir = ml_dir / "forecast_next_checkpoint"
    canopy_dir = ml_dir / "forecast_canopy"
    img_dir = ml_dir / "forecast_images"

    doc.add_paragraph("09-08-2026")
    doc.add_heading("9. Future Growth Forecasting (t → t+4)", level=1)
    doc.add_paragraph(
        "Sections 3–7 predict fresh weight at the same harvest checkpoint (nowcast). "
        "This section starts a true forecast module: given information available at DAT t, "
        "predict what happens at DAT t+4. Three steps are reported: (1) next-checkpoint "
        "fresh weight, (2) canopy size (mask area), and (3) Phase A future-image generation "
        "with a climate-conditioned Pix2Pix model (training underway / setup complete)."
    )

    # ---- 9.1 weight ----
    doc.add_heading("9.1 Next-Checkpoint Weight Forecast", level=2)
    doc.add_paragraph(
        "Pairs are built from consecutive harvest checkpoints four days apart. "
        "Features at day t (image / sensors / compact) predict fresh weight at day t+4. "
        "Baselines: Persist (same weight in 4 days) and GrowMean (add the mean 4-day gain "
        "from training folds). Script: ml/train_next_checkpoint_forecast.py. "
        "Outputs: ml/forecast_next_checkpoint/."
    )
    doc.add_paragraph(
        "Plant-level Trial 1: 68 pairs. Fair validation is leave-one-target-day-out "
        "(lodo_target_day). Leave-one-plant-out is also reported but is optimistic because "
        "the same harvest day can appear in both train and test. Tray-level: 14 pairs "
        "across trials; sensors ElasticNet is strongest on both tray splits."
    )

    wt_results = wt_dir / "results_next_checkpoint_forecast.csv"
    wt_summary = wt_dir / "forecast_summary.json"
    if wt_results.exists():
        rows = read_csv_rows(wt_results)

        def _pick_weight_table(level: str, split: str, variants: list[str]) -> list[dict]:
            out: list[dict] = []
            subset = [
                r
                for r in rows
                if r.get("level") == level and r.get("split") == split and r.get("variant") in variants
            ]
            # baselines once (from first variant present)
            for model in ("Persist", "GrowMean"):
                for r in subset:
                    if r.get("model") == model:
                        out.append(
                            {
                                "variant": "(baseline)",
                                "model": model,
                                "mae_g": fmt(r.get("mae_g", ""), 2),
                                "rmse_g": fmt(r.get("rmse_g", ""), 2),
                                "r2": fmt(r.get("r2", ""), 3),
                                "mae_vs_persist": fmt(r.get("mae_vs_persist", ""), 2),
                            }
                        )
                        break
            for variant in variants:
                vrows = [
                    r
                    for r in subset
                    if r.get("variant") == variant and r.get("model") not in ("Persist", "GrowMean")
                ]
                if not vrows:
                    continue
                best = min(vrows, key=lambda r: float(r["mae_g"]))
                out.append(
                    {
                        "variant": variant,
                        "model": best["model"],
                        "mae_g": fmt(best.get("mae_g", ""), 2),
                        "rmse_g": fmt(best.get("rmse_g", ""), 2),
                        "r2": fmt(best.get("r2", ""), 3),
                        "mae_vs_persist": fmt(best.get("mae_vs_persist", ""), 2),
                    }
                )
            return out

        add_table(
            doc,
            title="Plant-level fair split (leave-one-target-day-out; n = 68 pairs). Best model per feature set:",
            columns=["variant", "model", "mae_g", "rmse_g", "r2", "mae_vs_persist"],
            rows=_pick_weight_table("plant", "lodo_target_day", ["sensors", "image", "compact"]),
        )
        add_table(
            doc,
            title="Tray-level (leave-one-target-day-out; n = 14 pairs). Best model per feature set:",
            columns=["variant", "model", "mae_g", "rmse_g", "r2", "mae_vs_persist"],
            rows=_pick_weight_table("tray", "lodo_target_day", ["sensors", "image", "compact"]),
        )
    else:
        doc.add_paragraph(f"[Missing weight-forecast results: {wt_results}]")

    if wt_summary.exists():
        summary = json.loads(wt_summary.read_text(encoding="utf-8"))
        best = summary.get("best") or {}
        plant_fair = best.get("plant/lodo_target_day") or {}
        tray_fair = best.get("tray/lodo_target_day") or {}
        bullets = []
        if plant_fair:
            bullets.append(
                f"Best fair plant forecast: {plant_fair.get('variant')} {plant_fair.get('model')} "
                f"(MAE {float(plant_fair['mae_g']):.1f} g vs Persist {float(plant_fair['persist_mae_g']):.1f} g; "
                f"R² {float(plant_fair['r2']):.3f})."
            )
        if tray_fair:
            bullets.append(
                f"Best fair tray forecast: {tray_fair.get('variant')} {tray_fair.get('model')} "
                f"(MAE {float(tray_fair['mae_g']):.1f} g vs Persist {float(tray_fair['persist_mae_g']):.1f} g)."
            )
        bullets.append(
            "Sensors alone beat image/compact on the fair plant split; current weight and recent climate "
            "carry most of the 4-day signal."
        )
        bullets.append(
            "Plant leave-one-plant-out MAE ~3 g is not the headline number (same-day leakage). "
            "Use leave-one-target-day-out (~20.5 g) for claims."
        )
        doc.add_heading("9.1.1 Interpretation", level=3)
        for item in bullets:
            doc.add_paragraph(item, style="List Bullet")

    # ---- 9.2 canopy ----
    doc.add_heading("9.2 Canopy Size Forecast", level=2)
    doc.add_paragraph(
        "Same t → t+4 pairing, but the target is FastSAM mask area (pixels) instead of grams. "
        "Cup-level is the main result (same camera FOV within a cup). Tray leave-one-trial is weak "
        "because Trial 1 vs Trial 2 camera framing changed. Script: ml/train_canopy_forecast.py. "
        "Outputs: ml/forecast_canopy/."
    )

    canopy_results = canopy_dir / "results_canopy_forecast.csv"
    canopy_summary = canopy_dir / "canopy_forecast_summary.json"
    if canopy_results.exists():
        rows = read_csv_rows(canopy_results)
        mean_px = None
        if canopy_summary.exists():
            mean_px = json.loads(canopy_summary.read_text(encoding="utf-8")).get("cup_mean_target_px")

        def _canopy_rows(level: str, split: str, models: list[str]) -> list[dict]:
            out = []
            for model in models:
                hit = [
                    r
                    for r in rows
                    if r.get("level") == level and r.get("split") == split and r.get("model") == model
                ]
                if not hit:
                    continue
                r = hit[0]
                mae = float(r["mae"])
                row = {
                    "model": model,
                    "mae_px": fmt(r.get("mae", ""), 1),
                    "rmse_px": fmt(r.get("rmse", ""), 1),
                    "r2": fmt(r.get("r2", ""), 3),
                    "mae_vs_persist": fmt(r.get("mae_vs_persist", ""), 1),
                }
                if mean_px:
                    row["mae_pct_of_mean"] = fmt(100.0 * mae / float(mean_px), 1)
                out.append(row)
            return out

        cup_models = ["Persist", "GrowMean", "RandomForest", "GradBoost", "XGBoost", "ElasticNet", "Ridge"]
        cols = ["model", "mae_px", "rmse_px", "r2", "mae_vs_persist"]
        if mean_px:
            cols.append("mae_pct_of_mean")
        add_table(
            doc,
            title=(
                "Cup-level canopy forecast (leave-one-cup-out; n = 121 pairs"
                + (f"; mean target ≈ {float(mean_px):.0f} px" if mean_px else "")
                + "):"
            ),
            columns=cols,
            rows=_canopy_rows("cup", "loo_cup", cup_models),
        )
        add_table(
            doc,
            title="Cup-level fair day split (leave-one-target-day-out; n = 121):",
            columns=cols,
            rows=_canopy_rows("cup", "lodo_target_day", cup_models),
        )
    else:
        doc.add_paragraph(f"[Missing canopy-forecast results: {canopy_results}]")

    doc.add_heading("9.2.1 Interpretation", level=3)
    canopy_bullets = [
        "Best cup leave-one-cup-out: RandomForest MAE ≈ 4193 px (~10.7% of mean future mask area), "
        "beating Persist (≈ 5223 px).",
        "Best cup leave-one-target-day-out: RandomForest MAE ≈ 4012 px (~10.2% of mean), R² ≈ 0.81.",
        "Cross-trial cup / tray splits do not beat Persist reliably — FOV and lighting differ between trials.",
        "Outputs: ml/forecast_canopy/results_canopy_forecast.csv, cup_pairs_mask_area.csv.",
    ]
    for item in canopy_bullets:
        doc.add_paragraph(item, style="List Bullet")

    # ---- 9.3 images ----
    doc.add_heading("9.3 Future Image Generation (Pix2Pix Phase A)", level=2)
    doc.add_paragraph(
        "Goal: given a cup crop at day t plus the next-4-day climate summary, generate what the "
        "same cup should look like at day t+4. Phase A uses a conditional Pix2Pix (U-Net generator + "
        "PatchGAN discriminator) with climate tiled into extra input channels. "
        "Trial 1 FastSAM cup crops only; horizon +4 days. "
        "Scripts: ml/forecast_images/build_pix2pix_pairs.py, ml/forecast_images/train_pix2pix.py."
    )
    doc.add_paragraph(
        "Why Pix2Pix first (not SD/ControlNet yet): only 43 Trial-1 pairs where both days exist. "
        "A small paired translator is the right smoke-test before a heavier diffusion stack. "
        "SDXL + ControlNet remains a planned Phase B quality upgrade if Phase A proves the pairing."
    )

    norm_path = img_dir / "dataset" / "trial1_128" / "climate_norm.json"
    manifest_path = img_dir / "dataset" / "trial1_128" / "manifest.csv"
    split_rows = []
    if norm_path.exists():
        blob = json.loads(norm_path.read_text(encoding="utf-8"))
        counts = {"train": 0, "val": 0, "test": 0}
        if manifest_path.exists():
            for r in read_csv_rows(manifest_path):
                counts[r.get("split", "")] = counts.get(r.get("split", ""), 0) + 1
        for name, key in (("train", "train_cups"), ("val", "val_cups"), ("test", "test_cups")):
            cups = blob.get(key) or []
            split_rows.append(
                {
                    "split": name,
                    "cups": ", ".join(str(c) for c in cups),
                    "n_pairs": str(counts.get(name, "")),
                    "role": {
                        "train": "fit G/D; climate normalization",
                        "val": "select best.pt (lowest val L1)",
                        "test": "held-out once after training",
                    }[name],
                }
            )
    add_table(
        doc,
        title="Trial 1 Pix2Pix pairs split by cup_id (no cup appears in two splits; 128×128 crops):",
        columns=["split", "cups", "n_pairs", "role"],
        rows=split_rows
        or [
            {"split": "train", "cups": "1, 2, 4, 5, 6, 7, 8", "n_pairs": "25", "role": "fit G/D"},
            {"split": "val", "cups": "9, 11", "n_pairs": "13", "role": "select best.pt"},
            {"split": "test", "cups": "12, 14", "n_pairs": "5", "role": "held-out once"},
        ],
    )
    doc.add_paragraph(
        "Climate features (12): current and 4-day-mean T_air, RH, CO₂, EC, PPFD, DLI. "
        "Normalization statistics are computed from the train split only."
    )

    doc.add_heading("9.3.1 Training Setup", level=2)
    train_rows = [
        {"item": "Model", "value": "Pix2Pix (climate-conditioned; λ_L1 = 100)"},
        {"item": "Image size", "value": "128 × 128"},
        {"item": "Batch size", "value": "8"},
        {"item": "Epochs", "value": "300"},
        {"item": "Device", "value": "CUDA"},
    ]

    add_table(
        doc,
        title="Pix2Pix Phase A training configuration:",
        columns=["item", "value"],
        rows=train_rows,
    )

    doc.add_heading("9.3.2 Interpretation and Next Steps", level=3)
    for item in [
        "Dataset and split discipline are in place; climate is train-normalized only.",
        "Full 300-epoch training writes best.pt from val L1, then scores the 5 test pairs once.",
        "Inspect samples/*.png for visual growth plausibility; numeric L1/SSIM alone is not enough on n=5 test.",
        "After Phase A finishes: optional Phase B (SD/ControlNet), recover more cup-days, then append final "
        "image metrics / grids to this report (append-only).",
        "Outputs: ml/forecast_images/dataset/trial1_128/, ml/forecast_images/runs/pix2pix_t1_128_*/.",
    ]:
        doc.add_paragraph(item, style="List Bullet")


def add_pix2pix_trial_compare_section(doc: Document) -> None:
    """Section 9.4 — finished Pix2Pix Trial 1 vs Trial 2 comparison (append-only)."""
    img_dir = PROJECT_ROOT / "ml" / "forecast_images" / "runs"
    t1_dir = img_dir / "pix2pix_t1_128_20260908_105032"
    t2_dir = img_dir / "pix2pix_t2_128_20260908_130245"

    def _load_run(run_dir: Path) -> dict:
        out: dict = {"run": run_dir.name}
        cfg_path = run_dir / "config.json"
        tm_path = run_dir / "test_metrics.json"
        if cfg_path.exists():
            out.update(json.loads(cfg_path.read_text(encoding="utf-8")))
        if tm_path.exists():
            out.update(json.loads(tm_path.read_text(encoding="utf-8")))
        return out

    t1 = _load_run(t1_dir)
    t2 = _load_run(t2_dir)

    doc.add_paragraph("09-10-2026")
    doc.add_heading("9.4 Pix2Pix Results: Trial 1 vs Trial 2", level=2)
    doc.add_paragraph(
        "Phase A training finished for both trials under the same recipe "
        "(128×128, 300 epochs, batch size 8, λ_L1 = 100, climate-conditioned Pix2Pix, "
        "cup-level train/val/test split). Trial 2 was trained alone (not mixed with Trial 1) "
        "as a same-camera check with more pairs. Lower L1 is better; higher SSIM is better."
    )

    setup_rows = [
        {
            "item": "Train / val / test pairs",
            "trial1": f"{t1.get('n_train', 25)} / {t1.get('n_val', 13)} / {t1.get('n_test', 5)}",
            "trial2": f"{t2.get('n_train', 59)} / {t2.get('n_val', 15)} / {t2.get('n_test', 4)}",
        },
        {
            "item": "Cups (train / val / test)",
            "trial1": "1,2,4–8 / 9,11 / 12,14",
            "trial2": "1–6 / 7,8 / 9,12",
        },
    ]
    add_table(
        doc,
        title="Training setup comparison:",
        columns=["item", "trial1", "trial2"],
        rows=setup_rows,
    )

    def _f(d: dict, key: str, nd: int = 4) -> str:
        v = d.get(key)
        if v is None:
            return "—"
        try:
            return f"{float(v):.{nd}f}"
        except Exception:
            return str(v)

    metric_rows = [
        {
            "metric": "Best epoch (min val L1)",
            "trial1": str(t1.get("best_epoch", "—")),
            "trial2": str(t2.get("best_epoch", "—")),
            "winner": "—",
        },
        {
            "metric": "Best val L1 (↓ better)",
            "trial1": _f(t1, "best_val_l1"),
            "trial2": _f(t2, "best_val_l1"),
            "winner": "Trial 2",
        },
        {
            "metric": "Best val SSIM (↑ better)",
            "trial1": _f(t1, "best_val_ssim"),
            "trial2": _f(t2, "best_val_ssim"),
            "winner": "Trial 2",
        },
        {
            "metric": "Held-out test L1 (↓ better)",
            "trial1": _f(t1, "test_l1"),
            "trial2": _f(t2, "test_l1"),
            "winner": "Trial 2",
        },
        {
            "metric": "Held-out test SSIM (↑ better)",
            "trial1": _f(t1, "test_ssim"),
            "trial2": _f(t2, "test_ssim"),
            "winner": "Trial 2",
        },
    ]
    add_table(
        doc,
        title="Validation and held-out test metrics (best.pt selected by val L1):",
        columns=["metric", "trial1", "trial2", "winner"],
        rows=metric_rows,
    )

    add_figure(
        doc,
        t1_dir / "samples" / "test_grid.png",
        caption=(
            "Figure 9.2. Trial 1 held-out test grid (input | generated | real). "
            "Generated frames remain blurry / unstructured (test SSIM negative)."
        ),
        width_in=5.8,
    )
    add_figure(
        doc,
        t2_dir / "samples" / "test_grid.png",
        caption=(
            "Figure 9.3. Trial 2 held-out test grid (input | generated | real). "
            "Numerically better than Trial 1, but still far from sharp plant structure."
        ),
        width_in=5.8,
    )

    doc.add_heading("9.4.1 Interpretation", level=3)
    for item in [
        "Trial 2 beats Trial 1 on every reported metric (test L1 0.363 vs 0.448; test SSIM 0.369 vs −0.057), "
        "consistent with more train pairs (59 vs 25).",
        "Neither run produces usable future-plant photos yet: generated images stay soft / artifacted. "
        "Treat Phase A as a negative result on image quality, with Trial 2 the stronger of two weak runs.",
        "Next data levers: recover missing cup-days, combine trials carefully (FOV differs), then optional Phase B "
        "(e.g. ControlNet). Do not expect more epochs alone to fix n≈25–59.",
        "Outputs: ml/forecast_images/runs/pix2pix_t1_128_20260908_105032/, "
        "ml/forecast_images/runs/pix2pix_t2_128_20260908_130245/ "
        "(test_metrics.json, samples/test_grid.png).",
    ]:
        doc.add_paragraph(item, style="List Bullet")


def add_controlnet_trial_compare_section(doc: Document) -> None:
    """Section 9.5 — ControlNet Phase B Trial 1 vs Trial 2 (append-only)."""
    img_dir = PROJECT_ROOT / "ml" / "forecast_images" / "runs"
    t1_dir = img_dir / "controlnet_t1_256_20260910_112537"
    t2_dir = img_dir / "controlnet_t2_256_20260910_130342"
    # Pix2Pix counterparts for a same-split reference table
    p1_dir = img_dir / "pix2pix_t1_128_20260908_105032"
    p2_dir = img_dir / "pix2pix_t2_128_20260908_130245"

    def _load_run(run_dir: Path) -> dict:
        out: dict = {"run": run_dir.name}
        cfg_path = run_dir / "config.json"
        tm_path = run_dir / "test_metrics.json"
        if cfg_path.exists():
            out.update(json.loads(cfg_path.read_text(encoding="utf-8")))
        if tm_path.exists():
            out.update(json.loads(tm_path.read_text(encoding="utf-8")))
        return out

    def _f(d: dict, key: str, nd: int = 4) -> str:
        v = d.get(key)
        if v is None:
            return "—"
        try:
            return f"{float(v):.{nd}f}"
        except Exception:
            return str(v)

    t1 = _load_run(t1_dir)
    t2 = _load_run(t2_dir)
    p1 = _load_run(p1_dir)
    p2 = _load_run(p2_dir)

    doc.add_paragraph("09-11-2026")
    doc.add_heading("9.5 ControlNet Results: Trial 1 vs Trial 2", level=2)
    doc.add_paragraph(
        "Phase B uses pretrained Stable Diffusion v1.5 with a ControlNet fine-tuned on the same "
        "cup-level t → t+4 pairs as Pix2Pix. Images were preprocessed (RGBA→RGB on white, ExGR "
        "vegetation bbox crop with padding, resize to 256×256). Conditioning = day-t crop; target = "
        "day-t+4 crop; prompt includes a short climate summary. Only ControlNet weights were trained "
        "(SD frozen). Same cup train/val/test split as Section 9.4; 300 epochs. "
        "Lower L1 is better; higher SSIM is better."
    )

    setup_rows = [
        {
            "item": "Train / val / test pairs",
            "trial1": f"{t1.get('n_train', 25)} / {t1.get('n_val', 13)} / {t1.get('n_test', 5)}",
            "trial2": f"{t2.get('n_train', 59)} / {t2.get('n_val', 15)} / {t2.get('n_test', 4)}",
        },
        {
            "item": "Cups (train / val / test)",
            "trial1": "1,2,4–8 / 9,11 / 12,14",
            "trial2": "1–6 / 7,8 / 9,12",
        },
    ]
    add_table(
        doc,
        title="Training setup comparison:",
        columns=["item", "trial1", "trial2"],
        rows=setup_rows,
    )

    metric_rows = [
        {
            "metric": "Best step (min val L1)",
            "trial1": str(t1.get("best_step", "—")),
            "trial2": str(t2.get("best_step", "—")),
            "winner": "—",
        },
        {
            "metric": "Best val L1 (↓ better)",
            "trial1": _f(t1, "best_val_l1"),
            "trial2": _f(t2, "best_val_l1"),
            "winner": "Trial 1",
        },
        {
            "metric": "Best val SSIM (↑ better)",
            "trial1": _f(t1, "best_val_ssim"),
            "trial2": _f(t2, "best_val_ssim"),
            "winner": "Trial 1",
        },
        {
            "metric": "Held-out test L1 (↓ better)",
            "trial1": _f(t1, "test_l1"),
            "trial2": _f(t2, "test_l1"),
            "winner": "Trial 2",
        },
        {
            "metric": "Held-out test SSIM (↑ better)",
            "trial1": _f(t1, "test_ssim"),
            "trial2": _f(t2, "test_ssim"),
            "winner": "Trial 2",
        },
    ]
    add_table(
        doc,
        title="Validation and held-out test metrics (best ControlNet by val L1):",
        columns=["metric", "trial1", "trial2", "winner"],
        rows=metric_rows,
    )

    # Cross-method reference (same cups; different size/preprocess — interpret carefully)
    cross_rows = [
        {
            "setting": "Trial 1 test L1",
            "pix2pix_128": _f(p1, "test_l1"),
            "controlnet_256": _f(t1, "test_l1"),
        },
        {
            "setting": "Trial 1 test SSIM",
            "pix2pix_128": _f(p1, "test_ssim"),
            "controlnet_256": _f(t1, "test_ssim"),
        },
        {
            "setting": "Trial 2 test L1",
            "pix2pix_128": _f(p2, "test_l1"),
            "controlnet_256": _f(t2, "test_l1"),
        },
        {
            "setting": "Trial 2 test SSIM",
            "pix2pix_128": _f(p2, "test_ssim"),
            "controlnet_256": _f(t2, "test_ssim"),
        },
    ]
    add_table(
        doc,
        title=(
            "Reference vs Pix2Pix (same cup split; not a strict apples-to-apples score — "
            "Pix2Pix is 128px raw crops, ControlNet is 256px preprocessed):"
        ),
        columns=["setting", "pix2pix_128", "controlnet_256"],
        rows=cross_rows,
    )

    add_figure(
        doc,
        t1_dir / "samples" / "test_grid.png",
        caption=(
            "Figure 9.4. ControlNet Trial 1 held-out test grid (input | generated | real). "
            "Test SSIM remains near zero (~0.03)."
        ),
        width_in=5.8,
    )
    add_figure(
        doc,
        t2_dir / "samples" / "test_grid.png",
        caption=(
            "Figure 9.5. ControlNet Trial 2 held-out test grid (input | generated | real). "
            "Slightly better test L1/SSIM than Trial 1 ControlNet, still not sharp plant structure."
        ),
        width_in=5.8,
    )

    doc.add_heading("9.5.1 Interpretation", level=3)
    for item in [
        "Within ControlNet, Trial 2 wins held-out test (L1 0.765 vs 0.879; SSIM 0.093 vs 0.030), "
        "again consistent with more train pairs (59 vs 25).",
        "Phase B did not solve the image-forecast problem on this data: ControlNet test SSIM stays low "
        "(≪ Pix2Pix Trial 2’s 0.37 on its own 128px metric). Absolute L1/SSIM should not be over-compared "
        "across pipelines because resolution and preprocessing differ.",
        "Main takeaway for the meeting: a stronger pretrained model + preprocessing still fails to produce "
        "usable future-plant photos at n≈25–59 — supporting the need for more paired cup-days "
        "(recover missing days / Trial 3), not only a different architecture.",
        "Outputs: ml/forecast_images/runs/controlnet_t1_256_20260910_112537/, "
        "ml/forecast_images/runs/controlnet_t2_256_20260910_130342/ "
        "(test_metrics.json, samples/test_grid.png).",
    ]:
        doc.add_paragraph(item, style="List Bullet")


def add_aim1_update_section(doc: Document) -> None:
    """Section 10 — Aim 1: lock vanilla FastSAM + refresh plant FW ablation + T2 tray check."""
    plant_dir = PROJECT_ROOT / "ml" / "plant"
    ft_eval = (
        PROJECT_ROOT
        / "segmentation"
        / "research"
        / "runs"
        / "fastsam_ft_vanilla_hq"
        / "eval_test.json"
    )

    doc.add_paragraph("09-24-2026")
    doc.add_heading("10. Aim 1 Update: Segmentation Lock and Plant Fresh-Weight Ablation", level=1)
    doc.add_paragraph(
        "This section records the Aim 1 work after Sections 6–9: (i) attempted FastSAM fine-tuning "
        "against research GT masks, (ii) decision to lock zero-shot vanilla FastSAM for production "
        "cup features, (iii) a refreshed plant-level fresh-weight ablation on Trial 1, and "
        "(iv) a Trial 2 tray-aggregate sanity check (Trial 2 has no per-plant weights)."
    )

    doc.add_heading("10.1 FastSAM Fine-Tune Attempt (Abandoned)", level=2)
    doc.add_paragraph(
        "Inspired by FLAsH-style plant instance work, we tried dataset-specific FastSAM fine-tuning "
        "on research GT under segmentation/research/. Take: practical FastSAM cup pipeline. "
        "Skip: full FT as the production segmenter. Improve attempt: train against our GT and "
        "push lettuce IoU/Dice; if FT does not beat vanilla drafts, keep vanilla."
    )
    if ft_eval.exists():
        summary = json.loads(ft_eval.read_text(encoding="utf-8")).get("summary", {})
        vanilla_iou = float(summary.get("vanilla_draft_mean_iou_lettuce", float("nan")))
        ft_iou = float(summary.get("finetuned_mean_iou_lettuce", float("nan")))
        vanilla_dice = float(summary.get("vanilla_draft_mean_dice_lettuce", float("nan")))
        ft_dice = float(summary.get("finetuned_mean_dice_lettuce", float("nan")))
        n = int(summary.get("n", 0))
        add_table(
            doc,
            title=f"Test-set lettuce mask quality vs research GT (n={n} frames):",
            columns=["method", "mean_iou", "mean_dice"],
            rows=[
                {
                    "method": "Vanilla FastSAM drafts (production)",
                    "mean_iou": f"{vanilla_iou:.3f}",
                    "mean_dice": f"{vanilla_dice:.3f}",
                },
                {
                    "method": "Fine-tuned FastSAM (best.pt)",
                    "mean_iou": f"{ft_iou:.3f}",
                    "mean_dice": f"{ft_dice:.3f}",
                },
            ],
        )
        doc.add_paragraph(
            f"Fine-tuning reduced lettuce IoU by ~{vanilla_iou - ft_iou:.3f} versus vanilla drafts "
            f"(goal was IoU≥0.60 / Dice≥0.70). Likely bottleneck: noisy / incomplete GT rather than "
            "model capacity. We therefore lock vanilla FastSAM cup drafts as the Aim 1 segmenter "
            "and leave FT checkpoints under segmentation/research/ only."
        )
    else:
        doc.add_paragraph(
            "Fine-tuned FastSAM underperformed vanilla drafts on the research test split; "
            "vanilla FastSAM drafts are locked for Aim 1 feature extraction."
        )

    doc.add_heading("10.2 Plant Fresh-Weight Ablation (Trial 1, seed=42)", level=2)
    doc.add_paragraph(
        "After locking vanilla FastSAM, plant ML tables were re-exported from daily_cup_features "
        "and the full ablation grid was re-run: feature sets {sensors, image, compact, full} × "
        "models {MeanBaseline, Ridge, ElasticNet, SVR-RBF, RandomForest, GradBoost, XGBoost} × "
        "splits {LOO, LODO, LOPO}. Dataset size: 79 plant-days (Trial 1). "
        "Primary metric remains LODO plant MAE (g). LOO/LOPO are optimistic and reported only as context."
    )

    best_path = plant_dir / "ablation_aim1_plant_fw_best.csv"
    if best_path.exists():
        best_rows = read_csv_rows(best_path)
        lodo_rows = []
        for r in best_rows:
            if r.get("split") != "lodo":
                continue
            lodo_rows.append(
                {
                    "feature_set": r["feature_set"],
                    "best_model": r["model"],
                    "n_features": r.get("n_features", ""),
                    "mae_g": fmt(r["mae_g"], 2),
                    "rmse_g": fmt(r["rmse_g"], 2),
                    "r2": fmt(r["r2"], 3),
                    "day_median_mae_g": fmt(r.get("day_median_mae_g", ""), 2),
                }
            )
        lodo_rows = sorted(lodo_rows, key=lambda x: float(x["mae_g"]))
        add_table(
            doc,
            title="Best model per feature set — LODO (seed=42, single run):",
            columns=[
                "feature_set",
                "best_model",
                "n_features",
                "mae_g",
                "rmse_g",
                "r2",
                "day_median_mae_g",
            ],
            rows=lodo_rows,
        )

        # Compact LOO / LOPO for the LODO winner feature sets
        all_abl = read_csv_rows(plant_dir / "ablation_aim1_plant_fw.csv")
        ctx_rows = []
        for fset in ("full", "compact", "sensors", "image"):
            for split in ("loo", "lopo"):
                cand = [
                    r
                    for r in all_abl
                    if r.get("feature_set") == fset
                    and r.get("split") == split
                    and r.get("model") != "MeanBaseline"
                ]
                if not cand:
                    continue
                b = min(cand, key=lambda r: float(r["mae_g"]))
                ctx_rows.append(
                    {
                        "feature_set": fset,
                        "split": split,
                        "best_model": b["model"],
                        "mae_g": fmt(b["mae_g"], 2),
                        "r2": fmt(b["r2"], 3),
                    }
                )
        add_table(
            doc,
            title="Context only — best LOO / LOPO (optimistic; not for model selection):",
            columns=["feature_set", "split", "best_model", "mae_g", "r2"],
            rows=ctx_rows,
        )

    mean_sd_best = plant_dir / "ablation_aim1_plant_fw_mean_sd_best.csv"
    if mean_sd_best.exists():
        doc.add_paragraph(
            "Multi-seed mean ± SD (seeds 42–46) was also computed; LODO headline numbers:"
        )
        ms_rows = []
        for r in read_csv_rows(mean_sd_best):
            if r.get("split") != "lodo":
                continue
            ms_rows.append(
                {
                    "feature_set": r["feature_set"],
                    "best_model": r["model"],
                    "mae_g_mean_pm_sd": r.get("mae_g_mean_pm_sd", ""),
                    "r2_mean": fmt(r.get("r2_mean", ""), 3),
                    "r2_sd": fmt(r.get("r2_sd", ""), 3),
                }
            )
        ms_rows = sorted(
            ms_rows,
            key=lambda x: float(str(x["mae_g_mean_pm_sd"]).split("±")[0].strip()),
        )
        add_table(
            doc,
            title="Best model per feature set — LODO mean ± SD (5 seeds):",
            columns=["feature_set", "best_model", "mae_g_mean_pm_sd", "r2_mean", "r2_sd"],
            rows=ms_rows,
        )
    else:
        doc.add_paragraph(
            "Multi-seed mean ± SD is not yet available. A prior “5-seed” attempt wrote five identical "
            "logs (same MD5): the ablation shell script hard-coded --seed 42 and overwrote the same "
            "CSV each time, so SD would be zero and is not reported. Re-run with "
            "ml/run_aim1_plant_fw_multiseed.sh (seeds 42–46, per-seed CSVs + aggregate)."
        )

    doc.add_heading("10.3 Trial 2 Tray-Aggregate Check", level=2)
    doc.add_paragraph(
        "Trial 2 has tray-level fresh weights only (no Plant-ID weights). We trained XGBoost on "
        "Trial 1 plant rows, predicted each Trial 2 cup, then aggregated cup predictions to a "
        "tray mean/median and compared to observed Trial 2 tray medians (7 days)."
    )
    t2_sum = plant_dir / "t2_tray_aggregate_summary.csv"
    if t2_sum.exists():
        t2_rows = []
        for r in read_csv_rows(t2_sum):
            t2_rows.append(
                {
                    "feature_set": r["feature_set"],
                    "mae_pred_mean_g": fmt(r["mae_pred_mean_g"], 2),
                    "mae_pred_median_g": fmt(r["mae_pred_median_g"], 2),
                    "n_days": r.get("n_days", ""),
                }
            )
        t2_rows = sorted(t2_rows, key=lambda x: float(x["mae_pred_median_g"]))
        add_table(
            doc,
            title="Trial 2: cup→tray aggregate vs observed tray median (MAE, g):",
            columns=["feature_set", "mae_pred_mean_g", "mae_pred_median_g", "n_days"],
            rows=t2_rows,
        )
        doc.add_paragraph(
            "Best tray-aggregate MAE is ~2.6 g (sensors, median aggregate). This is a sanity check "
            "only: it collapses many cups and does not replace Trial 1 LODO plant MAE (~23 g) as "
            "the Aim 1 headline metric."
        )

    doc.add_heading("10.4 Interpretation and Next Steps", level=2)
    for item in [
        "Production segmenter for Aim 1: vanilla FastSAM cup drafts (no further FT planned unless GT quality improves).",
        "Headline plant FW: LODO XGBoost ≈ 23.2–23.6 g MAE on full/compact/sensors (seed=42); image-only weaker (~28.6 g).",
        "LOO MAE of ~3–5 g remains optimistic (same-day plants visible) and must not be quoted as the main result.",
        "Trial 2 tray check (~2.6 g) supports cross-trial transfer at tray scale but does not prove per-plant accuracy on Trial 2.",
        "Required next experiment: genuine 5-seed LODO mean±SD via ml/run_aim1_plant_fw_multiseed.sh "
        "(~35 min). Then optionally deepen multimodal fusion / cup tracking rather than more FT.",
        "Outputs: ml/plant/ablation_aim1_plant_fw*.csv, ml/plant/t2_tray_aggregate_*.csv, "
        "segmentation/research/runs/fastsam_ft_vanilla_hq/eval_test.json.",
    ]:
        doc.add_paragraph(item, style="List Bullet")


def add_aim2_update_section(doc: Document) -> None:
    """Section 11 — Aim 2 Phase A phenotype bridge + LOTO tray hard test."""
    tuned = PROJECT_ROOT / "ml" / "aim2_phenotype_tuned"
    loto = PROJECT_ROOT / "ml" / "aim2_loto"

    doc.add_paragraph("09-24-2026")
    doc.add_heading("11. Aim 2: Phenotype-Consistency Forecast (Phase A)", level=1)
    doc.add_paragraph(
        "Aim 2 here is not a new diffusion paper. Take (FGTD / Agricrafter / Drees): an intermediate "
        "future phenotype before biomass. Skip: retraining Pix2Pix/ControlNet in this phase. "
        "Improve: classical ML bridge under LODO and leave-one-trial-out, judged by FW MAE and "
        "canopy MAE, using sensors + FastSAM canopy features."
    )

    doc.add_heading("11.1 Phenotype Bridge (what it is)", level=2)
    doc.add_paragraph(
        "Two classical ML stages (not deep learning): (1) forecast canopy mask area at t+4 from "
        "features at t; (2) map that predicted canopy (+ sensors + current FW) to fresh weight at t+4. "
        "Tuned run selected RandomForest for canopy and ElasticNet for the FW bridge on plant LODO."
    )
    for item in [
        "Direct route: features@t → FW@t+4 (ElasticNet).",
        "Phenotype bridge: features@t → mask@t+4 (RF) → FW@t+4 (ElasticNet).",
        "Baselines: Persist (FW@t) and GrowMean (FW@t + mean 4-day gain from other folds).",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("11.2 Plant LODO Results (Trial 1, tuned, 5 seeds)", level=2)
    mean_sd = tuned / "results_aim2_mean_sd.csv"
    if mean_sd.exists():
        rows_in = read_csv_rows(mean_sd)
        # one row per route — if multiple models, keep lowest mae_g_mean
        by_route: dict[str, dict] = {}
        for r in rows_in:
            route = r.get("route", "")
            try:
                mae = float(r.get("mae_g_mean") or "nan")
            except Exception:
                mae = float("nan")
            prev = by_route.get(route)
            if prev is None or (
                mae == mae and (prev.get("_mae", float("inf")) != prev.get("_mae") or mae < prev["_mae"])
            ):
                rr = dict(r)
                rr["_mae"] = mae
                by_route[route] = rr
        table = []
        for route in ("persist", "grow_mean", "direct_fw", "phenotype_bridge", "canopy_only"):
            r = by_route.get(route)
            if not r:
                continue
            table.append(
                {
                    "route": route,
                    "model": r.get("model", ""),
                    "mae_g_mean_pm_sd": r.get("mae_g_mean_pm_sd", "—"),
                    "canopy_mae_px_mean_pm_sd": r.get("canopy_mae_px_mean_pm_sd", "—"),
                }
            )
        add_table(
            doc,
            title="Plant t→t+4 LODO (68 pairs, seeds 42–46):",
            columns=["route", "model", "mae_g_mean_pm_sd", "canopy_mae_px_mean_pm_sd"],
            rows=table,
        )
        doc.add_paragraph(
            "Headline: phenotype bridge FW MAE = 19.33 ± 1.18 g vs persist 30.59 g and direct "
            "20.82 ± 1.05 g. Canopy forecast beats persist on pixels (~8097 vs ~10136)."
        )

    doc.add_heading("11.3 Leave-One-Trial-Out Tray Test (hard generalization)", level=2)
    doc.add_paragraph(
        "Tray-level t→t+4 pairs (7 per trial). Train one trial, test the other. N=2 cycles only — "
        "honest hard test, not definitive. Camera FOV differs between trials, so canopy transfer is expected to struggle."
    )
    loto_csv = loto / "results_aim2_loto_mean_sd.csv"
    if loto_csv.exists():
        lrows = read_csv_rows(loto_csv)
        table = []
        for r in lrows:
            if r.get("route") not in ("persist", "grow_mean", "direct_fw", "phenotype_bridge"):
                continue
            table.append(
                {
                    "fold": r.get("fold", ""),
                    "route": r.get("route", ""),
                    "model": r.get("model", ""),
                    "mae_g_mean_pm_sd": r.get("mae_g_mean_pm_sd", "—"),
                    "canopy_mae_px_mean_pm_sd": r.get("canopy_mae_px_mean_pm_sd", "—"),
                }
            )
        add_table(
            doc,
            title="Tray LOTO t→t+4 (5 seeds):",
            columns=["fold", "route", "model", "mae_g_mean_pm_sd", "canopy_mae_px_mean_pm_sd"],
            rows=table,
        )
        doc.add_paragraph(
            "T1→T2: direct ElasticNet 11.72 g beats bridge 15.04 g; both beat persist 25.01 g. "
            "T2→T1: direct 5.74 g, bridge 6.46 g vs persist 30.73 g. "
            "Cross-trial canopy MAE is worse than persist (FOV / scale shift) — FW still transfers via sensors + current weight."
        )

    doc.add_heading("11.4 Interpretation and Next Steps", level=2)
    for item in [
        "Phase A success: within Trial 1, phenotype bridge is the best FW forecast among tested routes.",
        "Cross-trial: prefer direct FW (sensors + fw_t); do not rely on raw mask-area transfer without FOV normalization.",
        "Next (Phase B): score existing Pix2Pix/ControlNet generations with phenotype consistency "
        "(vegetation / mask-area agreement vs real t+4), not only L1/SSIM.",
        "Outputs: ml/aim2_phenotype_tuned/, ml/aim2_loto/.",
    ]:
        doc.add_paragraph(item, style="List Bullet")


def add_aim1_fw_refine_section(doc: Document) -> None:
    """Section 12 — Aim 1 FW methodology refine (curated packs, nested LODO, log target)."""
    refine = PROJECT_ROOT / "ml" / "plant" / "fw_refine"

    doc.add_paragraph("10-02-2026")
    doc.add_heading("12. Aim 1 FW Methodology Refine (Publishable Ablations)", level=1)
    doc.add_paragraph(
        "After locking vanilla FastSAM and the Section 10 compact/sensors LODO baseline "
        "(~23 g MAE), we ran a paper-facing Aim 1 refine on Trial 1 plant-days (n=79). "
        "Take (Lin 2022): compact canopy + climate trait fusion. Take (tomato FGTD 2026): "
        "classical/XGBoost models under sparse harvest labels. Take (Zhang 2026): "
        "physiology-lite extras (day², log mask area). Skip: Chen/Drees-scale image GANs as "
        "the FW estimator. Improve: curated low-dim packs, nested LODO model selection, "
        "optional log1p(FW), and explicit day-only / MeanBaseline controls so gains are not "
        "over-claimed."
    )

    doc.add_heading("12.1 Setup", level=2)
    for item in [
        "Source table: ml/plant/plant_checkpoint_trial1_compact.csv (vanilla FastSAM cup features + sensors).",
        "Feature packs: lin_phy (~41 curated canopy+climate), canopy (~49), climate (~25), compact (~166).",
        "Primary split: leave-one-harvest-day-out (LODO). LOO/LOPO recorded only as optimistic checks.",
        "Seeds 42–46. Models: Ridge, ElasticNet, RF, GradBoost, XGBoost (+ depth variants).",
        "Protocols: (i) fixed models; (ii) nested LODO model selection per held-out day; (iii) ± log1p(FW).",
        "Scripts: ml/refine_aim1_fw.py, ml/run_aim1_fw_refine.sh → ml/plant/fw_refine/.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("12.2 Fixed-Model LODO Results (5 seeds)", level=2)
    doc.add_paragraph(
        "Headline table uses the best model name chosen per seed when aggregating “Best:*” rows; "
        "for XGBoost we also report the fixed XGBoost row mean±SD across all five seeds. "
        "ElasticNet/Ridge are deterministic: identical MAE across seeds → SD = 0 (seed-invariant), "
        "not a failed multi-seed run."
    )

    fixed_path = refine / "fw_refine_summary_fixed.csv"
    raw_path = refine / "fw_refine_all_rows_fixed.csv"
    table_rows: list[dict] = []

    # Prefer explicit XGBoost compact + ElasticNet lin_phy log from raw for clean paper numbers
    if raw_path.exists():
        import statistics

        raw = read_csv_rows(raw_path)

        def seed_stats(pack: str, model: str, log_target: str) -> dict | None:
            vals = [
                float(r["mae_g"])
                for r in raw
                if r.get("pack") == pack
                and r.get("split") == "lodo"
                and r.get("model") == model
                and str(r.get("log_target")).lower() == log_target
            ]
            if not vals:
                return None
            mean = sum(vals) / len(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            r2s = [
                float(r["r2"])
                for r in raw
                if r.get("pack") == pack
                and r.get("split") == "lodo"
                and r.get("model") == model
                and str(r.get("log_target")).lower() == log_target
            ]
            nfeat = next(
                (
                    r.get("n_features", "")
                    for r in raw
                    if r.get("pack") == pack and r.get("model") == model
                ),
                "",
            )
            return {
                "pack": pack,
                "model": model,
                "log_target": log_target,
                "n_features": nfeat,
                "n_seeds": str(len(vals)),
                "mae_g_mean_pm_sd": f"{mean:.2f} ± {sd:.2f}",
                "r2_mean": f"{(sum(r2s)/len(r2s)):.3f}" if r2s else "",
                "note": "seed-invariant" if sd == 0.0 and model in ("ElasticNet", "Ridge", "MeanBaseline") else "",
            }

        for pack, model, log_t in [
            ("lin_phy", "ElasticNet", "true"),
            ("canopy", "ElasticNet", "true"),
            ("compact", "XGBoost", "false"),
            ("lin_phy", "XGBoost_d2", "false"),
            ("canopy", "XGBoost_d2", "false"),
            ("climate", "XGBoost", "false"),
            ("lin_phy", "MeanBaseline", "false"),
        ]:
            row = seed_stats(pack, model, log_t)
            if row:
                table_rows.append(row)

    if not table_rows and fixed_path.exists():
        for r in read_csv_rows(fixed_path):
            if r.get("model") == "MeanBaseline":
                continue
            table_rows.append(
                {
                    "pack": r["pack"],
                    "model": r["model"],
                    "log_target": r.get("log_target", ""),
                    "n_features": r.get("n_features", ""),
                    "n_seeds": r.get("n_seeds", ""),
                    "mae_g_mean_pm_sd": r.get("mae_pm", ""),
                    "r2_mean": fmt(r.get("r2_mean", ""), 3),
                    "note": "",
                }
            )

    if table_rows:
        add_table(
            doc,
            title="Fixed-model LODO (Trial 1, seeds 42–46) — selected paper rows:",
            columns=[
                "pack",
                "model",
                "log_target",
                "n_features",
                "n_seeds",
                "mae_g_mean_pm_sd",
                "r2_mean",
                "note",
            ],
            rows=table_rows,
        )

    doc.add_paragraph(
        "Sanity check (same LODO protocol): day-only Ridge on log1p(FW) achieves ~12.6 g MAE. "
        "Thus lin_phy ElasticNet log (10.33 g, seed-invariant) is largely a growth-curve + log-scale "
        "effect; plant canopy features add a modest gain over day-only (~2 g). "
        "MeanBaseline LODO remains ~66.7 g. The Section 10-comparable stochastic headline remains "
        "compact XGBoost on raw FW (~24.0 ± 1.13 g across five seeds; best single seed ~22.97 g)."
    )

    doc.add_heading("12.3 Nested LODO Selection", level=2)
    nested_path = refine / "fw_refine_summary_nested.csv"
    if nested_path.exists():
        nest_rows = []
        for r in read_csv_rows(nested_path):
            if "NestedSelect" not in str(r.get("model", "")):
                continue
            nest_rows.append(
                {
                    "pack": r["pack"],
                    "log_target": r.get("log_target", ""),
                    "n_features": r.get("n_features", ""),
                    "mae_g_mean_pm_sd": r.get("mae_pm", ""),
                    "r2_mean": fmt(r.get("r2_mean", ""), 3),
                }
            )
        nest_rows = sorted(
            nest_rows,
            key=lambda x: float(str(x["mae_g_mean_pm_sd"]).split("±")[0].strip()),
        )
        add_table(
            doc,
            title="Nested LODO model selection — mean ± SD over seeds:",
            columns=["pack", "log_target", "n_features", "mae_g_mean_pm_sd", "r2_mean"],
            rows=nest_rows,
        )
        doc.add_paragraph(
            "Nested selection did not beat a well-chosen fixed model on this n=79 setting and can "
            "inflate variance (e.g. compact NestedSelect ~25.3 ± 2.0 g vs fixed XGBoost ~24.0 ± 1.1 g). "
            "We keep nested results as a negative / cautionary ablation for the paper Methods."
        )
    else:
        doc.add_paragraph("Nested summary CSV not found under ml/plant/fw_refine/.")

    doc.add_heading("12.4 Interpretation (paper claims)", level=2)
    for item in [
        "Do not quote “10.33 ± 0.00” as multi-seed uncertainty: ElasticNet is seed-invariant; report as 10.33 g (deterministic) and show day-only (~12.6 g).",
        "For stochastic models, quote mean±SD (compact XGBoost ~24.0 ± 1.1 g LODO).",
        "Curated lin_phy is interpretable and competitive under log+linear models; compact+XGBoost remains the robust raw-FW baseline aligned with Section 10.",
        "Image-generation Aim 2 paths stay secondary for FW; this tabular refine is the publishable Aim 1 direction given sample size.",
        "Outputs: ml/plant/fw_refine/fw_refine_summary_fixed.csv, fw_refine_summary_nested.csv, lin_phy_features.json, fw_refine_run.log.",
    ]:
        doc.add_paragraph(item, style="List Bullet")


def add_future_fw_attribution_section(doc: Document) -> None:
    """Section 13 — future FW (t→t+4) LODO + factor attribution."""
    fw_dir = PROJECT_ROOT / "ml" / "future_fw_paper"
    attr_dir = PROJECT_ROOT / "ml" / "fw_attribution"
    next_dir = PROJECT_ROOT / "ml" / "forecast_next_checkpoint"

    doc.add_paragraph("10-02-2026")
    doc.add_heading("13. Future Fresh-Weight Forecast and Factor Attribution", level=1)
    doc.add_paragraph(
        "This section addresses unknown future weights (harvest at t+4 days) and which "
        "factors drive predictions. Take (FGTD / Aim 2 Phase A): dense sensors + sparse "
        "labels with a phenotype bridge and classical ML. Take (Lin): multimodal climate + "
        "canopy. Skip: image GANs as the FW estimator. Improve: multi-seed nested LODO for "
        "future FW, explicit Persist/GrowMean baselines, and publishable factor attribution "
        "(permutation importance + partial dependence; SHAP unavailable under the current "
        "NumPy/shap binary mismatch)."
    )

    doc.add_heading("13.1 Future-FW LODO (Trial 1, n=68 pairs, horizon=4 d)", level=2)
    doc.add_paragraph(
        "Leave-one-target-harvest-day-out on plant pairs (features @ t → FW @ t+4). "
        "Seeds 42–46 with nested model selection. Routes: Persist, GrowMean, direct FW "
        "(sensors + fw_t + prior gain), phenotype bridge (RF canopy@t+4 → ElasticNet FW)."
    )
    mean_sd = fw_dir / "results_aim2_mean_sd.csv"
    if mean_sd.exists():
        rows = []
        for r in read_csv_rows(mean_sd):
            if r.get("route") not in ("persist", "grow_mean", "direct_fw", "phenotype_bridge"):
                continue
            rows.append(
                {
                    "route": r["route"],
                    "model": r.get("model", ""),
                    "mae_g_mean_pm_sd": r.get("mae_g_mean_pm_sd", ""),
                    "r2_mean_pm_sd": r.get("r2_mean_pm_sd", ""),
                    "mae_vs_persist": r.get("mae_vs_persist_mean_pm_sd", ""),
                }
            )
        # sort by mae mean
        rows = sorted(
            rows,
            key=lambda x: float(str(x["mae_g_mean_pm_sd"]).split("±")[0].strip()),
        )
        add_table(
            doc,
            title="Future FW LODO mean ± SD (5 seeds):",
            columns=["route", "model", "mae_g_mean_pm_sd", "r2_mean_pm_sd", "mae_vs_persist"],
            rows=rows,
        )
        doc.add_paragraph(
            "Headline: phenotype bridge 19.33 ± 1.18 g beats direct ElasticNet 20.82 ± 1.05 g; "
            "both beat GrowMean 27.35 g and Persist 30.59 g (~11 g and ~10 g MAE reduction vs persist). "
            "This is the publishable future-weight result for Trial 1."
        )

    next_sum = next_dir / "forecast_summary.json"
    if next_sum.exists():
        blob = json.loads(next_sum.read_text(encoding="utf-8"))
        best = (blob.get("best") or {}).get("plant/lodo_target_day") or {}
        if best:
            doc.add_paragraph(
                f"Supporting next-checkpoint ablation (plant LODO): best variant={best.get('variant')} "
                f"model={best.get('model')} MAE={fmt(best.get('mae_g', ''), 2)} g "
                f"(persist {fmt(best.get('persist_mae_g', ''), 2)} g). "
                "LOO plant splits remain optimistic and are not used as the headline."
            )

    doc.add_heading("13.2 Factor Attribution", level=2)
    doc.add_paragraph(
        "Because tray climate is nearly constant across plants on a given harvest day, "
        "within-day LODO permutation cannot credit climate drivers. We therefore report "
        "(i) full-fit row-wise permutation importance and (ii) day-block permutation "
        "(shuffle whole harvest days) as explanatory rankings, while predictive skill stays "
        "the LODO table above. Figures: ml/fw_attribution/*_importance.png and *_partial_dependence.png."
    )

    for title, path in [
        (
            "Future FW (t→t+4) — top permutation importances (MAE increase if shuffled, g):",
            attr_dir / "forecast_permutation_importance.csv",
        ),
        (
            "Future FW — day-block importances (credits tray climate):",
            attr_dir / "forecast_dayblock_importance.csv",
        ),
        (
            "Nowcast FW — top permutation importances:",
            attr_dir / "nowcast_permutation_importance.csv",
        ),
    ]:
        if not path.exists():
            continue
        rows = []
        for r in read_csv_rows(path)[:8]:
            rows.append(
                {
                    "feature": r.get("feature", ""),
                    "mae_increase_mean": fmt(r.get("mae_increase_mean", ""), 2),
                    "mae_increase_sd": fmt(r.get("mae_increase_sd", ""), 2),
                }
            )
        add_table(doc, title=title, columns=["feature", "mae_increase_mean", "mae_increase_sd"], rows=rows)

    for fig, cap in [
        (
            attr_dir / "forecast_permutation_importance.png",
            "Figure: future-FW permutation importance (explanatory full fit).",
        ),
        (
            attr_dir / "forecast_partial_dependence.png",
            "Figure: future-FW partial dependence (EC, CO2, VPD, light, fw_t).",
        ),
        (
            attr_dir / "nowcast_dayblock_importance.png",
            "Figure: nowcast day-block importance (DAT dominates; then canopy area).",
        ),
    ]:
        if fig.exists():
            add_figure(doc, fig, cap, width_in=5.8)

    doc.add_heading("13.3 Interpretation (paper claims)", level=2)
    for item in [
        "Unknown future weights: phenotype bridge is best among tested routes (19.33 ± 1.18 g LODO MAE, 5 seeds).",
        "Actionable drivers for future FW (after current weight / prior gain / DAT): EC, night/day air temperature, VPD, humidity — candidate set-points for controlled experiments, not automatic closed-loop claims.",
        "Nowcast attribution: DAT/day dominates (growth stage); canopy mask area/coverage next — consistent with Lin-style canopy traits.",
        "Do not claim deficiency diagnosis from these rankings; they explain model sensitivity, not causal plant pathology.",
        "Outputs: ml/future_fw_paper/, ml/fw_attribution/, ml/forecast_next_checkpoint/, ml/future_fw_paper_pipeline.log.",
    ]:
        doc.add_paragraph(item, style="List Bullet")


def add_recommender_ar_section(doc: Document) -> None:
    """Section 14 — autoregressive FW + grower recommendation CLI (Trial 1 & 2)."""
    from datetime import date

    ar_root = PROJECT_ROOT / "ml" / "future_fw_ar"
    rec_dir = PROJECT_ROOT / "ml" / "recommendations"

    doc.add_paragraph(date.today().strftime("%m-%d-%Y"))
    doc.add_heading(
        "14. Autoregressive Future-FW and Grower Recommendation Layer",
        level=1,
    )
    doc.add_paragraph(
        "This section upgrades the future-weight route to match the requested "
        "autoregressive framing (learn change; optionally step day-to-day) and adds a "
        "grower-facing recommendation CLI with SHAP-grounded plain-English summaries. "
        "Take (professor note / FGTD-style forecasting): change models and multi-step roll. "
        "Skip: closed-loop RL and causal deficiency diagnosis. "
        "Improve vs Section 13 direct level model: AR-Δ (fw_t + predicted Δ) and daily roll "
        "t→t+1→…→t+4, plus Trial 2 transfer testing."
    )

    doc.add_heading("14.1 Why autoregressive (and the label constraint)", level=2)
    doc.add_paragraph(
        "Fresh weight is measured only every 4 days. Therefore we report two AR routes: "
        "(i) label-honest AR-Δ over the 4-day weigh-in step; "
        "(ii) daily Δ on log1p-interpolated mid-interval weights, then roll "
        "t→t+1→…→t+4 (secondary / optimistic because mid-days are synthetic). "
        "Primary predictive metric remains LODO MAE (g) on true weigh-in endpoints."
    )

    doc.add_heading("14.2 Trial 1 plant LODO (primary within-trial result)", level=2)
    t1p = ar_root / "trial1_plant" / "results_ar_mean_sd.csv"
    if not t1p.exists():
        t1p = ar_root / "results_ar_mean_sd.csv"
    if t1p.exists():
        rows = []
        for r in read_csv_rows(t1p):
            rows.append(
                {
                    "route": r.get("route", ""),
                    "mae_g_mean_pm_sd": r.get("mae_mean_pm_sd", ""),
                    "r2_mean": fmt(r.get("r2_mean", ""), 3),
                    "note": (
                        "secondary (interp labels)"
                        if r.get("route") == "ar_daily_roll4"
                        else "true weigh-in labels"
                    ),
                }
            )
        rows = sorted(
            rows,
            key=lambda x: float(str(x["mae_g_mean_pm_sd"]).split("±")[0].strip()),
        )
        add_table(
            doc,
            title="Trial 1 plant future-FW LODO (n=68 pairs, seeds 42–46):",
            columns=["route", "mae_g_mean_pm_sd", "r2_mean", "note"],
            rows=rows,
        )
        doc.add_paragraph(
            "Headline: AR-Δ (ar_delta_4d) reaches 9.07 g LODO MAE vs 17.26 g for the older "
            "direct level model and 30.59 g Persist. Daily roll is second (12.49 g) and is "
            "kept as a secondary timeline tool because mid-interval FW is interpolated."
        )

    doc.add_heading("14.3 Trial 2 (tray median) and T1→T2 transfer", level=2)
    doc.add_paragraph(
        "Trial 2 does not have per-plant fresh-weight labels in the plant checkpoint tables "
        "(only tray-median FW on harvest days). Checkpoint sensor window features for Trial 2 "
        "are largely empty, so tray climate for pairs is filled from daily drivers / daily "
        "sensor files. Within-Trial-2 LODO uses only ~7 pairs and is secondary; the primary "
        "Trial-2 generalization check is train Trial 1 plant → test Trial 2 tray."
    )

    t2 = ar_root / "trial2_tray" / "results_ar_mean_sd.csv"
    if t2.exists():
        rows = []
        for r in read_csv_rows(t2):
            rows.append(
                {
                    "route": r.get("route", ""),
                    "mae_g_mean_pm_sd": r.get("mae_mean_pm_sd", ""),
                    "r2_mean": fmt(r.get("r2_mean", ""), 3),
                }
            )
        rows = sorted(
            rows,
            key=lambda x: float(str(x["mae_g_mean_pm_sd"]).split("±")[0].strip()),
        )
        add_table(
            doc,
            title="Trial 2 tray LODO (n=7 pairs; small-n caveat):",
            columns=["route", "mae_g_mean_pm_sd", "r2_mean"],
            rows=rows,
        )

    trn = ar_root / "transfer_t1_to_t2" / "results_transfer_mean_sd.csv"
    if trn.exists():
        rows = []
        for r in read_csv_rows(trn):
            rows.append(
                {
                    "route": r.get("route", ""),
                    "mae_g_mean_pm_sd": r.get("mae_mean_pm_sd", ""),
                    "r2_mean": fmt(r.get("r2_mean", ""), 3),
                }
            )
        rows = sorted(
            rows,
            key=lambda x: float(str(x["mae_g_mean_pm_sd"]).split("±")[0].strip()),
        )
        add_table(
            doc,
            title="Primary Trial 2 check — train Trial 1 plant → test Trial 2 tray:",
            columns=["route", "mae_g_mean_pm_sd", "r2_mean"],
            rows=rows,
        )
        doc.add_paragraph(
            "Transfer headline: AR-Δ 16.41 g beats Persist 25.01 g and direct level 33.67 g "
            "on Trial 2 tray endpoints. This supports AR-Δ as the more transferable "
            "parameterization, while acknowledging tray-only labels and climate fill."
        )

    t1t = ar_root / "trial1_tray" / "results_ar_mean_sd.csv"
    if t1t.exists():
        rows = []
        for r in read_csv_rows(t1t):
            if r.get("route") not in ("ar_delta_4d", "direct_level", "persist", "ar_daily_roll4"):
                continue
            rows.append(
                {
                    "route": r.get("route", ""),
                    "mae_g_mean_pm_sd": r.get("mae_mean_pm_sd", ""),
                }
            )
        rows = sorted(
            rows,
            key=lambda x: float(str(x["mae_g_mean_pm_sd"]).split("±")[0].strip()),
        )
        add_table(
            doc,
            title="Trial 1 tray LODO (fair unit match to Trial 2; n=7):",
            columns=["route", "mae_g_mean_pm_sd"],
            rows=rows,
        )

    doc.add_heading("14.4 Grower recommendation CLI (decision support)", level=2)
    doc.add_paragraph(
        "CLI: ml/recommend_growth.py. Commands: status, what_if, reach_target, when, report. "
        "Primary +4d engine is now AR-Δ; timelines use daily AR roll when artifacts exist. "
        "Set-point suggestions stay inside observed Trial 1 p05–p95 ranges. "
        "Local SHAP (LinearExplainer) explains which factors help/slow the expected change; "
        "template NLP turns those numbers into a short plant-focused Summary "
        "(clearer wording, not higher numerical accuracy)."
    )
    for item in [
        "status — expected weight in ~4 days, likely range, helping/slowing factors, summary",
        "what_if — counterfactual EC / temperature / CO₂ / … changes",
        "reach_target — suggested tweaks for next weigh-in + days-to-goal (current vs suggested)",
        "when — days-to-goal under current settings only",
        "report — tray table for all plants on a day",
    ]:
        doc.add_paragraph(item, style="List Bullet")
    doc.add_paragraph(
        "Example (Trial 1 plant 3, day 12): AR-Δ expected ~49.3 g vs measured 49.8 g "
        "(much tighter than the old direct ~47 g ±17 g framing). "
        "Example (plant 5, day 16 → 100 g): day-by-day path and ~7-day timeline under current settings, "
        "with optimistic flag if a 4-day jump exceeds past growth."
    )
    if (rec_dir / "last_reach_target.txt").exists():
        doc.add_paragraph(
            "Latest CLI text snapshot saved under ml/recommendations/last_*.txt "
            "(human-readable; JSON also stored)."
        )

    doc.add_heading("14.5 Paper claims and limitations", level=2)
    for item in [
        "Primary within-Trial-1 future-FW route for the recommender: AR-Δ LODO MAE 9.07 g (vs Persist 30.59 g).",
        "Professor autoregressive request: implemented as AR-Δ (label-honest) + daily roll t→t+1→…→t+4 (secondary).",
        "Trial 2: no per-plant FW labels; primary T2 result is T1→T2 tray transfer (AR-Δ 16.41 g vs Persist 25.01 g).",
        "SHAP/NLP are explanatory layers for grower answers; they do not replace LODO metrics.",
        "Recommendations are decision support inside observed ranges — not automatic greenhouse control or proven causal effects.",
        "Outputs: ml/future_fw_ar/ (trial1_plant, trial1_tray, trial2_tray, transfer_t1_to_t2), ml/recommend_growth.py, ml/recommendations/.",
    ]:
        doc.add_paragraph(item, style="List Bullet")


def add_best_model_summary_section(doc: Document) -> None:
    """One comparison table of best model per feature set (8-row and 79-row)."""
    ml_dir = PROJECT_ROOT / "ml"
    plant_dir = ml_dir / "plant"

    def best_row(path: Path, extra: dict) -> dict:
        rows = [r for r in read_csv_rows(path) if r.get("model") != "MeanBaseline"]
        b = sorted(rows, key=lambda r: float(r["mae_g"]))[0]
        out = dict(extra)
        out["best_model"] = b["model"]
        out["n_features"] = b.get("n_features", "")
        out["mae_g"] = fmt(b["mae_g"], 2)
        out["r2"] = fmt(b["r2"], 3)
        return out

    doc.add_heading("6.6 Summary: Best Model by Feature Set", level=2)
    doc.add_paragraph(
        "The table below lists only the best model for each input set. "
        "The 8-row results are the tray-median leave-one-out numbers from Section 3. "
        "The 79-row results are the plant-level leave-one-harvest-day-out numbers from Section 6.3 "
        "(the fair plant-level split). Trial 2 8-row targets are the protocol-shifted Trial 1 medians."
    )

    table_rows = [
        best_row(
            ml_dir / "results_checkpoint_trial1_image_loo.csv",
            {"dataset": "Trial 1 tray-median (8 rows)", "feature_set": "Image-only", "split": "LOO"},
        ),
        best_row(
            ml_dir / "results_checkpoint_trial1_compact_loo.csv",
            {"dataset": "Trial 1 tray-median (8 rows)", "feature_set": "Compact", "split": "LOO"},
        ),
        best_row(
            ml_dir / "results_checkpoint_trial1_loo.csv",
            {"dataset": "Trial 1 tray-median (8 rows)", "feature_set": "Full", "split": "LOO"},
        ),
        best_row(
            ml_dir / "results_checkpoint_trial2_image_loo.csv",
            {"dataset": "Trial 2 tray-median (8 rows)", "feature_set": "Image-only", "split": "LOO"},
        ),
        best_row(
            ml_dir / "results_checkpoint_trial2_compact_loo.csv",
            {"dataset": "Trial 2 tray-median (8 rows)", "feature_set": "Compact", "split": "LOO"},
        ),
        best_row(
            ml_dir / "results_checkpoint_trial2_loo.csv",
            {"dataset": "Trial 2 tray-median (8 rows)", "feature_set": "Full", "split": "LOO"},
        ),
        best_row(
            plant_dir / "results_plant_checkpoint_trial1_image_lodo.csv",
            {"dataset": "Trial 1 plant-level (79 rows)", "feature_set": "Image-only", "split": "LODO"},
        ),
        best_row(
            plant_dir / "results_plant_checkpoint_trial1_sensors_lodo.csv",
            {"dataset": "Trial 1 plant-level (79 rows)", "feature_set": "Sensors-only", "split": "LODO"},
        ),
        best_row(
            plant_dir / "results_plant_checkpoint_trial1_compact_lodo.csv",
            {"dataset": "Trial 1 plant-level (79 rows)", "feature_set": "Compact", "split": "LODO"},
        ),
        best_row(
            plant_dir / "results_plant_checkpoint_trial1_lodo.csv",
            {"dataset": "Trial 1 plant-level (79 rows)", "feature_set": "Full", "split": "LODO"},
        ),
    ]
    add_table(
        doc,
        title="Best model per feature set:",
        columns=["dataset", "feature_set", "split", "best_model", "n_features", "mae_g", "r2"],
        rows=table_rows,
    )
    doc.add_paragraph(
        "MAE is in grams. Lower MAE and higher R² are better. "
        "LOO and LODO are different tests, so 8-row and 79-row MAE values are not strictly interchangeable; "
        "the table is a compact scoreboard, not a claim that the splits are identical."
    )


def main() -> None:
    """
    FULL rebuild of the progress report.

    Do NOT run this for routine updates. The user tracks dates and table layouts in the
    existing .docx; rewriting the whole file changes prior sections.

    For new content only, use:
      python3 append_to_progress_report.py --section early_warning

    Full rebuild requires an explicit flag:
      python3 generate_professor_report_docx.py --force-full-regen
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="FULL rebuild of Lettuce_Yield_Prediction_Progress_Report.docx (destructive to prior layout)."
    )
    parser.add_argument(
        "--force-full-regen",
        action="store_true",
        help="Required to overwrite the existing report. Prefer append_to_progress_report.py instead.",
    )
    args = parser.parse_args()
    if not args.force_full_regen:
        raise SystemExit(
            "Refusing to overwrite the progress report.\n"
            "Prior sections/dates/table layouts must be preserved.\n"
            "To ADD a section only:  python3 append_to_progress_report.py --section early_warning\n"
            "To fully rebuild (rare): python3 generate_professor_report_docx.py --force-full-regen"
        )

    out_path = PROJECT_ROOT / "Lettuce_Yield_Prediction_Progress_Report.docx"

    doc = Document()
    doc.add_heading("Lettuce Yield Prediction — Progress Report", 0)
    doc.add_paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ---------------------------------------------------------------------
    # Executive summary
    # ---------------------------------------------------------------------
    doc.add_heading("1. Executive Summary", level=1)
    doc.add_paragraph(
        "This report summarizes the completed image segmentation comparison, tray- and plant-level "
        "yield-prediction baselines, a mechanistic crop-model + residual hybrid ML stage, and an "
        "early-warning classifier for hydroponic lettuce. We compared three segmentation methods on "
        "matched timelapse frames, selected FastSAM for feature extraction, evaluated multiple "
        "regressors on harvest checkpoints, introduced a physics-based Van Henten / Farquhar crop "
        "model whose residuals are a target for image and sensor ML, and tested DAT ≤ 12 features "
        "for flagging plants that finish below the 227 g target (Section 8). Crop-model and ML "
        "metrics are reported separately in Section 7; trajectory overlays are in the figures."
    )

    # ---------------------------------------------------------------------
    # Segmentation methods + comparison
    # ---------------------------------------------------------------------
    doc.add_heading("2. Segmentation: Methods and Results", level=1)

    doc.add_heading("2.1 Methods Compared", level=2)
    doc.add_paragraph("We compared the following segmentation approaches (no dataset-specific fine-tuning):")
    for item in [
        "FastSAM (Fast Segment Anything): zero-shot instance segmentation optimized for speed.",
        "SAM (Segment Anything Model): zero-shot foundation-model segmentation.",
        "ExGR + Otsu + connected components: classical vegetation-index thresholding baseline.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("2.2 Evaluation Setup", level=2)
    doc.add_paragraph(
        "All three methods were evaluated on the same 53 matched frames (one frame per trial-day). "
        "Identical post-processing was applied to all methods (mask-size filtering and plant/pipe rejection heuristics) "
        "to ensure a fair comparison. Reported overlap metrics (Dice/IoU) quantify agreement between methods, not absolute "
        "accuracy against manual ground truth."
    )

    doc.add_heading("2.3 Quantitative Segmentation Results (Aggregate Means)", level=2)
    seg_rows = read_csv_rows(PROJECT_ROOT / "segmentation" / "comparison" / "metrics_summary.csv")
    # Normalize formatting
    for r in seg_rows:
        r["mean_time_sec"] = fmt(r.get("mean_time_sec", ""), 3)
        r["mean_raw_masks"] = fmt(r.get("mean_raw_masks", ""), 2)
        r["mean_plant_crops"] = fmt(r.get("mean_plant_crops", ""), 2)
        r["mean_plant_coverage_frac"] = fmt(r.get("mean_plant_coverage_frac", ""), 4)
        r["mean_plant_color_frac"] = fmt(r.get("mean_plant_color_frac", ""), 4)
        r["composite_score"] = fmt(r.get("composite_score", ""), 4)

    add_table(
        doc,
        title="Segmentation comparison summary:",
        columns=[
            "method",
            "n_images",
            "mean_time_sec",
            "mean_raw_masks",
            "mean_plant_crops",
            "mean_plant_coverage_frac",
            "mean_plant_color_frac",
            "composite_score",
        ],
        rows=seg_rows,
    )

    doc.add_heading("2.4 Pairwise Mask Agreement (Dice and IoU)", level=2)
    doc.add_paragraph(
        "Dice and IoU were computed on the union of accepted plant masks for each frame, comparing "
        "methods pairwise across the same 53 images. These values measure inter-method agreement "
        "(how similarly two methods label plant regions), not accuracy against manual ground truth."
    )
    pair_path = PROJECT_ROOT / "segmentation" / "comparison" / "pairwise_overlap_summary.csv"
    pair_rows = read_csv_rows(pair_path)
    for r in pair_rows:
        r["mean_dice"] = fmt(r.get("mean_dice", ""), 3)
        r["mean_iou"] = fmt(r.get("mean_iou", ""), 3)
    add_table(
        doc,
        title="Mean Dice and IoU between segmentation methods:",
        columns=["method_pair", "mean_dice", "mean_iou", "n_frames"],
        rows=pair_rows,
    )
    doc.add_paragraph(
        "FastSAM and SAM show substantially higher agreement with each other (Dice 0.603, IoU 0.452) "
        "than either does with the classical ExGR baseline (IoU below 0.11 for both pairs)."
    )

    doc.add_heading("2.5 Segmentation Conclusion", level=2)
    doc.add_paragraph(
        "FastSAM was selected for downstream yield prediction because it achieved the highest canopy coverage "
        "while maintaining strong agreement with SAM and substantially faster runtime than SAM. "
        "ExGR was kept as a classical baseline but captured lower canopy coverage in dense tray imagery."
    )

    # ---------------------------------------------------------------------
    # Prediction results
    # ---------------------------------------------------------------------
    doc.add_heading("3. Yield Prediction: Models and Results", level=1)
    doc.add_paragraph(
        "We evaluated multiple regressors using leave-one-out (LOO) validation on 8 harvest checkpoints per trial. "
        "Results are exploratory due to small sample size."
    )
    doc.add_paragraph(
        "Three input feature sets were compared: image-only (FastSAM canopy size, color, and window growth); "
        "compact image + sensors (those image features plus window-average climate, nutrient, and actuator signals); "
        "and the full feature set (the same sources with window min/max/std statistics in addition to means)."
    )

    def add_prediction_section(trial: str) -> None:
        doc.add_heading(f"3.{1 if trial=='trial1' else 2} Results for {trial.title()}", level=2)

        variants = [
            ("Image-only features", f"results_checkpoint_{trial}_image_loo.csv"),
            ("Compact (image + sensors)", f"results_checkpoint_{trial}_compact_loo.csv"),
            ("Full feature set", f"results_checkpoint_{trial}_loo.csv"),
        ]

        for label, fname in variants:
            p = PROJECT_ROOT / "ml" / fname
            rows = read_csv_rows(p)
            rows_sorted = sorted(rows, key=lambda r: float(r["mae_g"]))
            best = rows_sorted[0]

            doc.add_heading(label, level=3)
            doc.add_paragraph(
                f"Best model: {best['model']} "
                f"(MAE={float(best['mae_g']):.2f} g, R²={float(best['r2']):.3f})."
            )

            table_rows = []
            for r in rows_sorted:
                table_rows.append({
                    "model": r["model"],
                    "n_features": r.get("n_features", ""),
                    "mae_g": fmt(r.get("mae_g", ""), 2),
                    "rmse_g": fmt(r.get("rmse_g", ""), 2),
                    "r2": fmt(r.get("r2", ""), 3),
                })

            add_table(
                doc,
                title=f"Leave-one-out validation results ({label.lower()}):",
                columns=["model", "n_features", "mae_g", "rmse_g", "r2"],
                rows=table_rows,
            )

    add_prediction_section("trial1")
    add_prediction_section("trial2")

    doc.add_heading("4. Selected Configuration for Next Phase", level=1)
    doc.add_paragraph(
        "For direct ML yield prediction (Sections 3–6), we recommend the compact feature set "
        "(image + sensors) as the primary configuration, with image-only as an ablation and FastSAM "
        "as the segmentation method. For the physics + ML path (Section 7), keep the crop model as "
        "the tray-level baseline and report hybrid residual ML with compact features, using LODO within "
        "a trial and Trial-1→Trial-2 hold-out as the generalization check. For early warning (Section 8), "
        "report leave-one-plant-out results at the 10% below-target threshold (where Trial 1 has "
        "positive labels), alongside the official 20% definition and tray-level DAT-12 rules."
    )

    doc.add_heading("5. Notes / Limitations", level=1)
    for item in [
        "Only 8 labeled harvest checkpoints are available per trial; metrics are therefore high-variance and should be interpreted as exploratory.",
        "Segmentation overlap metrics (Dice/IoU) reported here measure agreement between methods; manual ground-truth masks are required for absolute segmentation accuracy (mIoU/Dice vs ground truth).",
        "Trial 2 labels may be protocol-shifted relative to Trial 1; cross-trial comparisons should be interpreted cautiously.",
        "Crop-model absolute grams depend on assumed bench area, dry-matter fraction (4.5%), and reconstructed PPFD (not logged). See Section 7.",
        "Early-warning (Section 8): Trial 1 has no plants ≥20% below 227 g, so the official Aim-3 label cannot yet be scored at plant level; 10%/15% thresholds are sensitivity checks with 2 and 1 positives.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    add_plant_level_section(doc)

    out_paths = [
        out_path,
        PROJECT_ROOT / "Initial plan" / "Lettuce_Yield_Prediction_Progress_Report.docx",
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        doc.save(p)
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()


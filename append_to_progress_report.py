"""
Append new sections to the existing progress report WITHOUT rewriting prior content.

Preserves dates, wording, and table layouts already in the .docx.

Usage:
  python3 append_to_progress_report.py --section early_warning
  python3 append_to_progress_report.py --section growth_forecast
  python3 append_to_progress_report.py --section pix2pix_trial_compare
  python3 append_to_progress_report.py --section aim1_fw_refine
  python3 append_to_progress_report.py --section aim2_update
  python3 append_to_progress_report.py --section recommender_ar

Never use generate_professor_report_docx.py for routine updates (that rebuilds everything).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document

from generate_professor_report_docx import (
    add_aim1_fw_refine_section,
    add_aim1_update_section,
    add_aim2_update_section,
    add_controlnet_trial_compare_section,
    add_early_warning_section,
    add_future_fw_attribution_section,
    add_growth_forecast_section,
    add_pix2pix_trial_compare_section,
    add_recommender_ar_section,
)

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT = PROJECT_ROOT / "Lettuce_Yield_Prediction_Progress_Report.docx"
REPORT_INITIAL = PROJECT_ROOT / "Initial plan" / "Lettuce_Yield_Prediction_Progress_Report.docx"

SECTION_MARKERS = {
    "early_warning": "8. Early-Warning Classification",
    "growth_forecast": "9. Future Growth Forecasting",
    "pix2pix_trial_compare": "9.4 Pix2Pix Results: Trial 1 vs Trial 2",
    "controlnet_trial_compare": "9.5 ControlNet Results: Trial 1 vs Trial 2",
    "aim1_update": "10. Aim 1 Update: Segmentation Lock and Plant Fresh-Weight Ablation",
    "aim2_update": "11. Aim 2: Phenotype-Consistency Forecast (Phase A)",
    "aim1_fw_refine": "12. Aim 1 FW Methodology Refine",
    "future_fw_attribution": "13. Future Fresh-Weight Forecast and Factor Attribution",
    "recommender_ar": "14. Autoregressive Future-FW and Grower Recommendation Layer",
}

SECTION_BUILDERS = {
    "early_warning": add_early_warning_section,
    "growth_forecast": add_growth_forecast_section,
    "pix2pix_trial_compare": add_pix2pix_trial_compare_section,
    "controlnet_trial_compare": add_controlnet_trial_compare_section,
    "aim1_update": add_aim1_update_section,
    "aim2_update": add_aim2_update_section,
    "aim1_fw_refine": add_aim1_fw_refine_section,
    "future_fw_attribution": add_future_fw_attribution_section,
    "recommender_ar": add_recommender_ar_section,
}


def already_has_section(doc: Document, marker: str) -> bool:
    for p in doc.paragraphs:
        if marker in (p.text or ""):
            return True
    return False


def append_section(path: Path, section: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing report: {path}")

    marker = SECTION_MARKERS[section]
    doc = Document(str(path))
    if already_has_section(doc, marker):
        print(f"Skip {path.name}: already contains '{marker}' (no changes).")
        return

    SECTION_BUILDERS[section](doc)
    doc.save(str(path))
    print(f"Appended '{section}' to {path} (prior sections untouched).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Append-only progress report updates.")
    parser.add_argument(
        "--section",
        required=True,
        choices=sorted(SECTION_MARKERS),
        help="Section to append if missing.",
    )
    parser.add_argument(
        "--also-initial-plan",
        action="store_true",
        help="Also append to Initial plan/ copy if that file exists.",
    )
    args = parser.parse_args()

    append_section(REPORT, args.section)
    if args.also_initial_plan and REPORT_INITIAL.exists():
        append_section(REPORT_INITIAL, args.section)


if __name__ == "__main__":
    main()

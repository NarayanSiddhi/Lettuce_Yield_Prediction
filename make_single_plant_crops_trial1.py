from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return x1, y1, x2, y2


def score_crop(crop: np.ndarray) -> float:
    """
    Pick the best frame of the day for the fixed plant ROI.
    Prefer sharper + better contrast + some green content.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 35, 30])
    upper_green = np.array([95, 255, 255])
    gmask = cv2.inRange(hsv, lower_green, upper_green)
    green_frac = float(np.count_nonzero(gmask)) / float(gmask.size)

    brightness = float(gray.mean())
    # Penalize very dark/night frames; reward usable daylight visibility.
    visibility = max(0.0, brightness - 55.0)
    return 0.45 * sharpness + 0.20 * contrast + 180.0 * green_frac + 1.1 * visibility


def extract_hour_from_filename(path: Path):
    """
    Extract hour from names like plant-YYYYMMDD-HHMMSS.jpg.
    Returns int hour or None if not parseable.
    """
    stem = path.stem
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    time_part = parts[-1]
    if len(time_part) < 2 or not time_part[:2].isdigit():
        return None
    return int(time_part[:2])


def main() -> None:
    project_root = Path(__file__).resolve().parent

    trial_dir = (
        project_root
        / "Timelapse_growtent"
        / "Trial-1-Camera Capture-20260323T192552Z-3-001"
        / "Trial-1-Camera Capture"
    )
    if not trial_dir.exists():
        raise FileNotFoundError(f"Missing trial directory: {trial_dir}")

    # Exact user-provided black box coordinates (x1, y1, x2, y2),
    # applied identically to every day.
    box_x1, box_y1, box_x2, box_y2 = 1721, 673, 3418, 2002

    out_dir = project_root / "crops" / "trial1_singleplant"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = project_root / "crops" / "trial1_singleplant_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    day_idx = 1

    date_dirs = sorted([p for p in trial_dir.iterdir() if p.is_dir()])
    if not date_dirs:
        raise RuntimeError("No date folders found.")

    for date_dir in date_dirs:
        date_str = date_dir.name
        jpg_files = sorted(date_dir.glob("*.jpg"))
        if not jpg_files:
            continue

        best_path = None
        best_score = -1.0
        best_crop = None
        best_xy = None

        for img_path in jpg_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            x1, y1, x2, y2 = clamp_box(box_x1, box_y1, box_x2, box_y2, w, h)
            crop = img[y1:y2, x1:x2]
            score = score_crop(crop)
            hour = extract_hour_from_filename(img_path)
            # Strong preference for daytime captures where plant is visible.
            if hour is not None:
                if 8 <= hour <= 18:
                    score += 80.0
                elif hour <= 5 or hour >= 21:
                    score -= 80.0
            if score > best_score:
                best_score = score
                best_path = img_path
                best_crop = crop
                best_xy = (x1, y1, x2, y2)

        if best_crop is None or best_path is None or best_xy is None:
            continue

        crop_path = out_dir / f"{date_str}.png"
        cv2.imwrite(str(crop_path), best_crop)

        # Save a quick preview image with ROI box for auditability.
        src = cv2.imread(str(best_path))
        x1, y1, x2, y2 = best_xy
        cv2.rectangle(src, (x1, y1), (x2, y2), (0, 255, 255), 6)
        cv2.imwrite(str(preview_dir / f"{date_str}_boxed.jpg"), src)

        rows.append(
            {
                "date": date_str,
                "day_index": day_idx,
                "crop_path": str(crop_path.relative_to(project_root)).replace("\\", "/"),
                "source_image": str(best_path.relative_to(project_root)).replace("\\", "/"),
                "selection_score": round(best_score, 4),
                "crop_x1": x1,
                "crop_y1": y1,
                "crop_x2": x2,
                "crop_y2": y2,
            }
        )
        day_idx += 1

    if not rows:
        raise RuntimeError("No crops generated.")

    df = pd.DataFrame(rows)
    csv_path = project_root / "image_daily_index_trial1_singleplant.csv"
    df.to_csv(csv_path, index=False)

    print(f"Saved {len(rows)} single-plant crops to {out_dir}")
    print(f"Saved ROI previews to {preview_dir}")
    print(f"Wrote index: {csv_path}")


if __name__ == "__main__":
    main()


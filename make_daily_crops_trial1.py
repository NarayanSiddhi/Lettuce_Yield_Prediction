from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def detect_center_pot_anchor(reference_img: np.ndarray):
    """
    Detect the net cup closest to image center in a reference frame.
    Returns anchor center (x, y). Falls back to image center.
    """
    h, w = reference_img.shape[:2]
    cx, cy = w // 2, h // 2

    gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=120,
        param1=100,
        param2=25,
        minRadius=25,
        maxRadius=90,
    )

    if circles is None:
        return cx, cy

    circles = np.round(circles[0]).astype(int)
    # Prefer an upper-center cup (more stable in this camera angle) if available.
    upper = [c for c in circles if c[1] <= cy]
    candidates = upper if upper else circles
    best = min(candidates, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)
    return int(best[0]), int(best[1])


def crop_from_anchor(img: np.ndarray, anchor_x: int, anchor_y: int, side: int = 700):
    """Fixed-size square crop centered on detected anchor."""
    h, w = img.shape[:2]
    half = side // 2
    x1 = max(anchor_x - half, 0)
    y1 = max(anchor_y - half, 0)
    x2 = min(anchor_x + half, w)
    y2 = min(anchor_y + half, h)
    crop = img[y1:y2, x1:x2]
    return crop, x1, y1, x2, y2


def score_crop_quality(crop: np.ndarray) -> float:
    """
    Score visibility quality for selecting best frame of the day.
    Higher sharpness/contrast and some green signal are preferred.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = float(gray.std())

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_frac = float(np.count_nonzero(green_mask)) / float(green_mask.size)

    return 0.7 * sharpness + 0.2 * contrast + 220.0 * green_frac


def choose_best_image_and_crop(jpg_files, anchor_x: int, anchor_y: int):
    """Pick best frame of the day based on crop quality around center pot."""
    best_path = None
    best_score = -1.0
    best_crop_data = None

    for p in jpg_files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        crop, x1, y1, x2, y2 = crop_from_anchor(img, anchor_x, anchor_y, side=700)
        score = score_crop_quality(crop)
        if score > best_score:
            best_score = score
            best_path = p
            best_crop_data = (crop, x1, y1, x2, y2)

    return best_path, best_score, best_crop_data


def main() -> None:
    """
    Create one cropped image per day for Trial-1 and an index CSV.

    For each day folder:
    - Detects a center-most net-cup anchor from a reference frame.
    - Crops around that same anchor region for every candidate image.
    - Selects the best frame among 16 based on crop clarity/visibility.
    - Saves crop under crops/trial1/<date>.png.
    - Writes image_daily_index_trial1.csv with metadata.
    """

    project_root = Path(__file__).resolve().parent

    trial_dir = (
        project_root
        / "Timelapse_growtent"
        / "Trial-1-Camera Capture-20260323T192552Z-3-001"
        / "Trial-1-Camera Capture"
    )

    if not trial_dir.exists():
        raise FileNotFoundError(f"Trial-1 directory not found: {trial_dir}")

    out_dir = project_root / "crops" / "trial1"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    day_idx = 1

    date_dirs = sorted([p for p in trial_dir.iterdir() if p.is_dir()])
    if not date_dirs:
        raise RuntimeError(f"No date folders found under {trial_dir}")

    # Build anchor from the latest day's noon frame when possible.
    ref_day = date_dirs[-1]
    ref_noon = sorted(ref_day.glob("*-120501.jpg"))
    ref_img_path = ref_noon[0] if ref_noon else sorted(ref_day.glob("*.jpg"))[0]
    ref_img = cv2.imread(str(ref_img_path))
    if ref_img is None:
        raise RuntimeError(f"Failed to read reference frame: {ref_img_path}")
    anchor_x, anchor_y = detect_center_pot_anchor(ref_img)

    # Single-plant lock: keep one consistent center-ish pot across all days.
    # If auto-anchor drifts, this fixed offset keeps tracking on one plant.
    anchor_x = int(anchor_x + 110)
    anchor_y = int(anchor_y - 90)

    for date_dir in date_dirs:
        date_str = date_dir.name  # e.g. 2025-10-14

        jpg_files = sorted(date_dir.glob("*.jpg"))
        if not jpg_files:
            continue

        best_img_path, best_score, best_crop_data = choose_best_image_and_crop(
            jpg_files, anchor_x, anchor_y
        )
        if best_img_path is None or best_crop_data is None:
            continue

        crop, x1, y1, x2, y2 = best_crop_data

        out_path = out_dir / f"{date_str}.png"
        cv2.imwrite(str(out_path), crop)

        rows.append(
            {
                "date": date_str,
                "day_index": day_idx,
                "crop_path": str(out_path.relative_to(project_root)).replace("\\", "/"),
                "source_image": str(best_img_path.relative_to(project_root)).replace("\\", "/"),
                "crop_x1": x1,
                "crop_y1": y1,
                "crop_x2": x2,
                "crop_y2": y2,
                "selection_score": round(float(best_score), 4),
                "anchor_x": anchor_x,
                "anchor_y": anchor_y,
            }
        )
        day_idx += 1

    if not rows:
        raise RuntimeError("No crops were created – please check the folder structure and images.")

    df = pd.DataFrame(rows)
    csv_path = project_root / "image_daily_index_trial1.csv"
    df.to_csv(csv_path, index=False)

    print(f"Saved {len(rows)} crops to {out_dir}")
    print(f"Index CSV written to {csv_path}")


if __name__ == "__main__":
    main()



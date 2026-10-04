from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def compute_features(img: np.ndarray) -> dict:
    """
    Compute robust image features for lettuce growth modeling.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Plant mask (green + red lettuce tones)
    green_mask = cv2.inRange(hsv, (25, 30, 25), (95, 255, 255))
    red1 = cv2.inRange(hsv, (0, 35, 25), (15, 255, 255))
    red2 = cv2.inRange(hsv, (160, 35, 25), (179, 255, 255))
    plant_mask = cv2.bitwise_or(green_mask, cv2.bitwise_or(red1, red2))

    kernel = np.ones((5, 5), np.uint8)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel)

    total_px = plant_mask.size
    plant_px = int(np.count_nonzero(plant_mask))
    green_px = int(np.count_nonzero(green_mask))

    green_ratio = green_px / total_px if total_px else 0.0
    vegetation_ratio = plant_px / total_px if total_px else 0.0
    canopy_area_px = plant_px

    if plant_px > 0:
        h_vals = hsv[:, :, 0][plant_mask > 0]
        s_vals = hsv[:, :, 1][plant_mask > 0]
        v_vals = hsv[:, :, 2][plant_mask > 0]
        mean_h = float(np.mean(h_vals))
        mean_s = float(np.mean(s_vals))
        mean_v = float(np.mean(v_vals))
        color_std = float(np.std(v_vals))
    else:
        mean_h = float(np.mean(hsv[:, :, 0]))
        mean_s = float(np.mean(hsv[:, :, 1]))
        mean_v = float(np.mean(hsv[:, :, 2]))
        color_std = float(np.std(hsv[:, :, 2]))

    # Blur/sharpness score
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Optional texture-like features
    edges = cv2.Canny(gray, 80, 160)
    texture_edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    texture_laplacian_mean_abs = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_64F))))

    return {
        "green_ratio": round(green_ratio, 6),
        "vegetation_ratio": round(vegetation_ratio, 6),
        "canopy_area_px": canopy_area_px,
        "mean_h": round(mean_h, 4),
        "mean_s": round(mean_s, 4),
        "mean_v": round(mean_v, 4),
        "color_std": round(color_std, 4),
        "blur_score": round(blur_score, 4),
        "texture_edge_density": round(texture_edge_density, 6),
        "texture_laplacian_mean_abs": round(texture_laplacian_mean_abs, 6),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parent
    crops_dir = project_root / "crops" / "trial1_singleplant"
    if not crops_dir.exists():
        raise FileNotFoundError(f"Missing crops folder: {crops_dir}")

    rows = []
    for img_path in sorted(crops_dir.glob("*.png")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        date_str = img_path.stem
        f = compute_features(img)
        rows.append({"date": date_str, "image_path": str(img_path.relative_to(project_root)).replace("\\", "/"), **f})

    if not rows:
        raise RuntimeError("No images found or all unreadable in crops/trial1_singleplant.")

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out_csv = project_root / "image_features_trial1_singleplant.csv"
    df.to_csv(out_csv, index=False)

    print(f"Saved {len(df)} rows to {out_csv}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()


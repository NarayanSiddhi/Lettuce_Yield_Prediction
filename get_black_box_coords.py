from pathlib import Path

import cv2


def main() -> None:
    """
    Helper script to capture the exact black-box crop coordinates.

    Instructions:
    - When the window opens, FIRST click the TOP-LEFT corner of the box you want.
    - Then click the BOTTOM-RIGHT corner of the same box.
    - Press any key to close the window.
    - The script will print BOX_COORDS: x1 y1 x2 y2 in the terminal.
    """

    # Use a representative frame from Trial-1 (you can change this path if needed)
    img_path = Path(
        "Timelapse_growtent/Trial-1-Camera Capture-20260323T192552Z-3-001/"
        "Trial-1-Camera Capture/2025-10-18/plant-20251018-120501.jpg"
    )
    if not img_path.exists():
        raise FileNotFoundError(f"Reference image not found: {img_path}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    display = img.copy()
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        nonlocal display, points
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) >= 2:
                return
            points.append((x, y))
            cv2.circle(display, (x, y), 8, (0, 0, 255), -1)
            cv2.imshow("Select black box corners", display)

    cv2.namedWindow("Select black box corners", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select black box corners", on_mouse)

    print(
        "Click TOP-LEFT corner of your desired black box, then BOTTOM-RIGHT corner, "
        "then press any key in the image window."
    )
    cv2.imshow("Select black box corners", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 2:
        print(f"Need exactly 2 clicks, got {len(points)}: {points}")
        return

    (x1, y1), (x2, y2) = points
    # Normalize so x1<x2, y1<y2
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    print(f"BOX_COORDS: {x1} {y1} {x2} {y2}")


if __name__ == "__main__":
    main()


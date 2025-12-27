import cv2
import numpy as np
from pathlib import Path
import argparse
from skimage.metrics import structural_similarity as ssim


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="Path to baseline image")
    p.add_argument("--current", required=True, help="Path to current image")
    p.add_argument("--out", required=True, help="Output image path")
    p.add_argument("--threshold", type=int, default=50, help="Difference threshold (0-255)")
    p.add_argument("--min-area", type=int, default=500, help="Minimum area for a change region")
    p.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel size (odd number)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline)
    current_path = Path(args.current)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load images
    img1 = cv2.imread(str(baseline_path))
    img2 = cv2.imread(str(current_path))
    if img1 is None or img2 is None:
        raise ValueError("Could not load images")

    # Ensure same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Convert to grayscale for SSIM
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    if args.blur > 0:
        gray1 = cv2.GaussianBlur(gray1, (args.blur, args.blur), 0)
        gray2 = cv2.GaussianBlur(gray2, (args.blur, args.blur), 0)

    # Compute SSIM
    (score, diff_ssim) = ssim(gray1, gray2, full=True)
    diff_ssim = (diff_ssim * 255).astype("uint8")

    # Threshold the SSIM difference
    _, thresh = cv2.threshold(diff_ssim, args.threshold, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around changes
    result = cv2.imread(str(current_path))  # Use original current
    if result.shape != img2.shape:
        result = cv2.resize(result, (img2.shape[1], img2.shape[0]))
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > args.min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(result, "CHANGED", (x, max(0, y - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Save result
    cv2.imwrite(str(out_path), result)
    print(f"Saved image diff: {out_path}, SSIM score: {score:.4f}")


if __name__ == "__main__":
    main()
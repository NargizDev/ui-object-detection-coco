from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-images", required=True, help="Directory with baseline images")
    p.add_argument("--current-images", required=True, help="Directory with current images")
    p.add_argument("--model", choices=["yolo", "fasterrcnn"], required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out-dir", required=True, help="Output directory for results")
    p.add_argument("--conf", type=float, default=0.1)
    return p.parse_args()


def run_cmd(cmd: str):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise Exception(f"Command failed: {cmd}")
    return result.stdout


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Predict baseline
    baseline_pred = out_dir / "baseline_pred.json"
    run_cmd(f"python inference/predict.py --model yolo --weights runs/detect/train2/weights/best.pt --images {args.baseline_images} --out {baseline_pred} --conf {args.conf}")

    # Predict current
    current_pred = out_dir / "current_pred.json"
    run_cmd(f"python inference/predict.py --model yolo --weights runs/detect/train2/weights/best.pt --images {args.current_images} --out {current_pred} --conf {args.conf}")

    # Postprocess baseline
    baseline_hier = out_dir / "baseline_hierarchical.json"
    run_cmd(f"python inference/postprocess.py --pred {baseline_pred} --out {baseline_hier}")

    # Postprocess current
    current_hier = out_dir / "current_hierarchical.json"
    run_cmd(f"python inference/postprocess.py --pred {current_pred} --out {current_hier}")

    # Compare
    diff = out_dir / "diff.json"
    run_cmd(f"python inference/compare.py --baseline {baseline_hier} --current {current_hier} --out {diff}")

    # Image diff for visual changes
    baseline_img_dir = Path(args.baseline_images)
    current_img_dir = Path(args.current_images)
    vis_diff_dir = out_dir / "vis_diff"
    vis_diff_dir.mkdir(parents=True, exist_ok=True)
    for current_img in current_img_dir.iterdir():
        if current_img.is_file() and current_img.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            baseline_img = baseline_img_dir / current_img.name
            if baseline_img.exists():
                out_img = vis_diff_dir / current_img.name
                run_cmd(f"python inference/image_diff.py --baseline {baseline_img} --current {current_img} --out {out_img}")

    # Generate report
    report = out_dir / "report.md"
    run_cmd(f"python inference/report.py --diff {diff} --out {report}")

    print(f"Analysis complete. Results in {out_dir}")
    print(f"Diff: {diff}")
    print(f"Visual diff: {vis_diff_dir}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()

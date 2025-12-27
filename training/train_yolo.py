from __future__ import annotations

import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to Ultralytics data yaml, e.g. data/processed/yolo/ui.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO model name or path to .pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="cpu", help="cpu, 0, 0,1 etc")
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name", default="train")
    p.add_argument(
        "--plots",
        choices=["auto", "on", "off"],
        default="auto",
        help="Plot metrics (auto disables when polars is unavailable or broken).",
    )
    return p.parse_args()

def resolve_plots_setting(choice: str) -> bool:
    if choice == "off":
        return False
    if choice == "on":
        return True
    try:
        import polars as pl
    except Exception:
        return False
    return hasattr(pl, "PyDataFrame")

def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise SystemExit("ultralytics is not installed. Run: pip install ultralytics") from e

    plots = resolve_plots_setting(args.plots)
    if args.plots == "auto" and not plots:
        print("Plotting disabled: polars missing or broken. Install polars or pass --plots on to force.")

    model = YOLO(args.model)
    model.train(
        data=str(Path(args.data)),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=plots,
    )
    print("OK")

if __name__ == "__main__":
    main()

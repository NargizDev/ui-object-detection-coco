from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["yolo", "fasterrcnn"], required=True)
    p.add_argument("--weights", required=True, help="Path to YOLO .pt or FasterR-CNN checkpoint .pt")
    p.add_argument("--images", required=True, help="Directory with images to run prediction on")
    p.add_argument("--out", required=True, help="Output COCO predictions json")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def coco_pred_record(image_id: int, category_id: int, bbox_xywh: List[float], score: float) -> Dict[str, Any]:
    return {"image_id": image_id, "category_id": category_id, "bbox": bbox_xywh, "score": float(score)}


def main() -> None:
    args = parse_args()
    img_dir = Path(args.images)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Map file_name -> incremental image_id based on sorted filenames
    image_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    file_to_id = {p.name: i + 1 for i, p in enumerate(image_files)}

    preds: List[Dict[str, Any]] = []

    if args.model == "yolo":
        from ultralytics import YOLO

        model = YOLO(args.weights)
        for p in image_files:
            results = model.predict(source=str(p), conf=args.conf, iou=args.iou, device=args.device, verbose=False)
            r = results[0]
            boxes = r.boxes
            if boxes is None:
                continue

            # Ultralytics gives cls in 0..K-1, but we don't know COCO category_id mapping here.
            # We'll store category_id = cls+1 by default.
            for b in boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = xyxy
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                cls = int(b.cls[0].item())
                score = float(b.conf[0].item())
                preds.append(coco_pred_record(file_to_id[p.name], cls + 1, bbox, score))

    else:
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        device = torch.device(args.device)
        ckpt = torch.load(args.weights, map_location=device)
        categories = ckpt["categories"]
        cls_to_cat_id = {int(k): int(v) for k, v in ckpt["cls_to_cat_id"].items()}

        num_classes = 1 + len(categories)
        model = fasterrcnn_resnet50_fpn_v2(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        with torch.no_grad():
            for p in image_files:
                img = Image.open(p).convert("RGB")
                w, h = img.size
                arr = np.asarray(img).astype(np.float32) / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1).to(device)

                out = model([t])[0]
                boxes = out["boxes"].cpu().numpy()
                labels = out["labels"].cpu().numpy()
                scores = out["scores"].cpu().numpy()

                for (x1, y1, x2, y2), lab, sc in zip(boxes, labels, scores):
                    if float(sc) < args.conf:
                        continue
                    cat_id = int(cls_to_cat_id.get(int(lab), int(lab)))
                    bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    preds.append(coco_pred_record(file_to_id[p.name], cat_id, bbox, float(sc)))

    out_path.write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved predictions: {out_path}")

if __name__ == "__main__":
    main()

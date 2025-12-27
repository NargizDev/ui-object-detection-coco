from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from ui_detect.utils.coco import (
    coco_to_class_index,
    categories_sorted,
    images_sorted,
    index_by_image_id,
    load_coco,
    save_coco,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--coco", required=True, help="Path to COCO instances json from CVAT export")
    p.add_argument("--images", required=True, help="Directory with the image files referenced in COCO (cv1.png...)")
    p.add_argument("--out", required=True, help="Output directory (processed dataset)")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy-images", action="store_true", help="Copy images into processed/yolo/images/* (safer)")
    return p.parse_args()

def bbox_coco_to_yolo(bbox_xywh: Tuple[float, float, float, float], w: int, h: int) -> Tuple[float, float, float, float]:
    x, y, bw, bh = bbox_xywh
    xc = x + bw / 2.0
    yc = y + bh / 2.0
    return (xc / w, yc / h, bw / w, bh / h)

def subset_coco(coco: Dict, image_ids: List[int]) -> Dict:
    image_ids_set = set(image_ids)
    images = [img for img in coco["images"] if int(img["id"]) in image_ids_set]
    anns = [ann for ann in coco["annotations"] if int(ann["image_id"]) in image_ids_set]
    return {
        "licenses": coco.get("licenses", []),
        "info": coco.get("info", {}),
        "categories": coco["categories"],
        "images": images,
        "annotations": anns,
    }

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    coco = load_coco(args.coco)
    cats = categories_sorted(coco)
    imgs = images_sorted(coco)
    ann_by_img = index_by_image_id(coco)
    cat2cls = coco_to_class_index(coco)

    out_root = Path(args.out)
    out_coco = out_root / "coco"
    out_yolo = out_root / "yolo"
    (out_coco).mkdir(parents=True, exist_ok=True)
    (out_yolo).mkdir(parents=True, exist_ok=True)

    # Split
    all_ids = [img.id for img in imgs]
    random.shuffle(all_ids)
    cut = max(1, int(len(all_ids) * args.train_ratio))
    train_ids = sorted(all_ids[:cut])
    val_ids = sorted(all_ids[cut:]) if len(all_ids) > 1 else sorted(all_ids)

    train_coco = subset_coco(coco, train_ids)
    val_coco = subset_coco(coco, val_ids)

    save_coco(train_coco, out_coco / "train.json")
    save_coco(val_coco, out_coco / "val.json")

    # YOLO dirs
    for split in ["train", "val"]:
        (out_yolo / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_yolo / "labels" / split).mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images)
    if not images_dir.exists():
        raise SystemExit(f"Images dir not found: {images_dir}")

    # Write YOLO labels
    def process_split(image_ids: List[int], split: str) -> None:
        for img in imgs:
            if img.id not in image_ids:
                continue
            src_img_path = images_dir / img.file_name
            if not src_img_path.exists():
                raise SystemExit(f"Missing image referenced by COCO: {src_img_path}")

            dst_img_path = out_yolo / "images" / split / img.file_name
            if args.copy_images:
                shutil.copy2(src_img_path, dst_img_path)
            else:
                # keep a lightweight copy anyway (Windows-safe)
                shutil.copy2(src_img_path, dst_img_path)

            label_path = out_yolo / "labels" / split / (Path(img.file_name).stem + ".txt")
            lines: List[str] = []
            for ann in ann_by_img.get(img.id, []):
                cls = cat2cls[int(ann["category_id"])]
                x, y, bw, bh = bbox_coco_to_yolo(tuple(ann["bbox"]), img.width, img.height)
                # clamp just in case
                x = min(max(x, 0.0), 1.0)
                y = min(max(y, 0.0), 1.0)
                bw = min(max(bw, 0.0), 1.0)
                bh = min(max(bh, 0.0), 1.0)
                lines.append(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")
            label_path.write_text("\n".join(lines), encoding="utf-8")

    process_split(train_ids, "train")
    process_split(val_ids, "val")

    # Ultralytics data yaml
    names = [c.name for c in cats]
    yaml_text = f"""path: {out_yolo.as_posix()}
train: images/train
val: images/val

names:
"""
    for i, name in enumerate(names):
        yaml_text += f"  {i}: {name}\n"
    (out_yolo / "ui.yaml").write_text(yaml_text, encoding="utf-8")

    # Stats
    counts = Counter()
    counts_by_split = {"train": Counter(), "val": Counter()}
    img_counts = {"train": len(train_ids), "val": len(val_ids)}
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        for img_id in ids:
            for ann in ann_by_img.get(img_id, []):
                cat_id = int(ann["category_id"])
                name = next(c.name for c in cats if c.id == cat_id)
                counts[name] += 1
                counts_by_split[split][name] += 1

    stats = {
        "num_images_total": len(imgs),
        "num_images": img_counts,
        "num_categories": len(cats),
        "categories": [{ "id": c.id, "name": c.name } for c in cats],
        "objects_total": sum(counts.values()),
        "objects_by_class": dict(counts),
        "objects_by_split": {k: dict(v) for k, v in counts_by_split.items()},
        "note": "If you have only ~10 images, treat training as pipeline demo, not as final quality measurement.",
    }
    (out_root / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OK")
    print(f"COCO train: {out_coco / 'train.json'}")
    print(f"COCO val:   {out_coco / 'val.json'}")
    print(f"YOLO data:  {out_yolo / 'ui.yaml'}")
    print(f"Stats:      {out_root / 'dataset_stats.json'}")

if __name__ == "__main__":
    main()

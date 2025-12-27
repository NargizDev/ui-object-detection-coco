from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="Directory with images")
    p.add_argument("--pred", required=True, help="Hierarchical predictions json")
    p.add_argument("--out-dir", required=True, help="Where to save visualized images")
    p.add_argument("--min-score", type=float, default=0.25)
    p.add_argument("--diff", help="Diff json to highlight changed elements in red")
    p.add_argument("--diff-only", action="store_true", help="Draw only changed elements from diff")
    return p.parse_args()


def draw_tree(img, tree: List[Dict], level=0, min_score=0.25):
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    color = colors[level % len(colors)]
    count = 0
    for node in tree:
        score = float(node.get("score", 0.0))
        if score < min_score:
            continue
        x, y, w, h = node["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        label = int(node["category_id"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label}:{score:.2f} L{level}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        count += 1
        if 'children' in node:
            child_count = draw_tree(img, node['children'], level + 1, min_score)
            count += child_count
    return count


def draw_changed(img, changed: List[Dict], min_score=0.25):
    for item in changed:
        if isinstance(item, dict) and 'baseline' in item:  # changed has baseline and current
            element = item['current']
        else:
            element = item
        score = float(element.get("score", 0.0))
        if score < min_score:
            continue
        x, y, w, h = element["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red, thicker
        cv2.putText(img, "CHANGED", (x1, max(0, y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


def main() -> None:
    args = parse_args()
    img_dir = Path(args.images)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hierarchical: Dict[str, List[Dict]] = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    diff_data = None
    if args.diff:
        diff_data: Dict[str, Dict] = json.loads(Path(args.diff).read_text(encoding="utf-8"))

    image_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    for idx, img_path in enumerate(image_files, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        tree = hierarchical.get(str(idx), [])
        count = 0
        if not args.diff_only:
            count = draw_tree(img, tree, min_score=args.min_score)
        if diff_data and str(idx) in diff_data:
            changed = diff_data[str(idx)].get('changed', [])
            draw_changed(img, changed, min_score=args.min_score)
            count += len([c for c in changed if float(c.get('current', c).get("score", 0.0)) >= args.min_score])
        print(f"Image {idx} ({img_path.name}): drew {count} elements")

        cv2.imwrite(str(out_dir / img_path.name), img)

    print(f"Saved visualizations: {out_dir}")


if __name__ == "__main__":
    main()

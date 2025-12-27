from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="COCO predictions json")
    p.add_argument("--out", required=True, help="Output hierarchical json")
    return p.parse_args()


def area(bbox: List[float]) -> float:
    return bbox[2] * bbox[3]


def contains(parent_bbox: List[float], child_bbox: List[float]) -> bool:
    px, py, pw, ph = parent_bbox
    cx, cy, cw, ch = child_bbox
    return px <= cx and py <= cy and px + pw >= cx + cw and py + ph >= cy + ch


def build_hierarchy(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Sort by area descending
    sorted_preds = sorted(predictions, key=lambda p: area(p['bbox']), reverse=True)
    tree = []
    for pred in sorted_preds:
        pred['children'] = []
        # Add semantic group
        cat_id = pred['category_id']
        if cat_id in [1, 2, 3, 4]:  # button, input, dropdown, checkbox
            pred['group'] = 'interactive'
        elif cat_id == 5:  # icon
            pred['group'] = 'icon'
        elif cat_id == 6:  # image
            pred['group'] = 'media'
        elif cat_id == 7:  # text
            pred['group'] = 'text'
        elif cat_id in [8, 9]:  # card, navbar
            pred['group'] = 'layout'
        else:
            pred['group'] = 'other'
        # Find parent
        parent = None
        for node in tree:
            if contains(node['bbox'], pred['bbox']):
                parent = node
                break
        if parent:
            parent['children'].append(pred)
        else:
            tree.append(pred)
    return tree


def image_key(pred: Dict[str, Any]) -> str:
    file_name = pred.get("file_name")
    if file_name:
        return str(file_name)
    image_id = pred.get("image_id")
    return str(image_id)


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    preds: List[Dict[str, Any]] = json.loads(pred_path.read_text(encoding="utf-8"))

    # Group by file_name when present, otherwise by image_id.
    by_image: Dict[str, List[Dict[str, Any]]] = {}
    for p in preds:
        by_image.setdefault(image_key(p), []).append(p)

    hierarchical = {}
    for img_key, img_preds in by_image.items():
        hierarchical[img_key] = build_hierarchy(img_preds)

    out_path.write_text(json.dumps(hierarchical, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved hierarchical predictions: {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset, Features, Sequence, Value, Image as HFImage
from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_id", required=True, help="e.g. username/ui-screenshots-coco")
    p.add_argument("--images", required=True, help="Directory with images")
    p.add_argument("--coco", required=True, help="COCO instances json")
    p.add_argument("--private", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images)
    coco = json.loads(Path(args.coco).read_text(encoding="utf-8"))

    cats = sorted(coco["categories"], key=lambda x: int(x["id"]))
    cat_id_to_name = {int(c["id"]): str(c["name"]) for c in cats}

    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    rows = []
    for img in sorted(coco["images"], key=lambda x: int(x["id"])):
        img_id = int(img["id"])
        path = images_dir / img["file_name"]
        if not path.exists():
            # skip missing images
            continue

        objects = []
        for ann in anns_by_img.get(img_id, []):
            objects.append({
                "bbox": [float(x) for x in ann["bbox"]],  # x,y,w,h
                "category_id": int(ann["category_id"]),
                "label": cat_id_to_name[int(ann["category_id"])],
            })

        rows.append({
            "image_id": img_id,
            "image": str(path),
            "objects": objects,
        })

    features = Features({
        "image_id": Value("int32"),
        "image": HFImage(),
        "objects": Sequence({
            "bbox": Sequence(Value("float32"), length=4),
            "category_id": Value("int32"),
            "label": Value("string"),
        }),
    })

    ds = Dataset.from_list(rows, features=features)

    api = HfApi()
    api.create_repo(repo_id=args.dataset_id, repo_type="dataset", private=bool(args.private), exist_ok=True)

    ds.push_to_hub(args.dataset_id)
    print(f"Uploaded: {args.dataset_id}")

if __name__ == "__main__":
    main()

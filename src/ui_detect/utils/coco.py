from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class CocoCategory:
    id: int
    name: str


@dataclass(frozen=True)
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class CocoAnnotation:
    id: int
    image_id: int
    category_id: int
    bbox: Tuple[float, float, float, float]  # x, y, w, h
    area: float | None
    iscrowd: int


def load_coco(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_coco(obj: Dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def index_by_image_id(coco: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    idx: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        idx.setdefault(int(ann["image_id"]), []).append(ann)
    return idx


def categories_sorted(coco: Dict[str, Any]) -> List[CocoCategory]:
    cats = [CocoCategory(int(c["id"]), str(c["name"])) for c in coco.get("categories", [])]
    return sorted(cats, key=lambda c: c.id)


def images_sorted(coco: Dict[str, Any]) -> List[CocoImage]:
    imgs = [
        CocoImage(int(i["id"]), str(i["file_name"]), int(i["width"]), int(i["height"]))
        for i in coco.get("images", [])
    ]
    return sorted(imgs, key=lambda i: i.id)


def coco_to_class_index(coco: Dict[str, Any]) -> Dict[int, int]:
    """Map COCO category_id -> 0..(K-1) in sorted order by category_id."""
    cats = categories_sorted(coco)
    return {c.id: idx for idx, c in enumerate(cats)}


def class_index_to_coco(coco: Dict[str, Any]) -> Dict[int, int]:
    """Map 0..(K-1) -> COCO category_id."""
    cats = categories_sorted(coco)
    return {idx: c.id for idx, c in enumerate(cats)}

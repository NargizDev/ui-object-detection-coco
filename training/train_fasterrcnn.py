from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Path to COCO train.json")
    p.add_argument("--val", required=True, help="Path to COCO val.json")
    p.add_argument("--images", required=True, help="Directory with all images referenced by COCO")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="outputs/fasterrcnn/model.pt")
    return p.parse_args()


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(coco: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]], Dict[int, int], List[Dict[str, Any]]]:
    images = sorted(coco["images"], key=lambda x: int(x["id"]))
    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    categories = sorted(coco["categories"], key=lambda x: int(x["id"]))
    cat_id_to_cls = {int(cat["id"]): i + 1 for i, cat in enumerate(categories)}  # +1 for background=0
    cls_to_cat_id = {v: k for k, v in cat_id_to_cls.items()}
    return images, anns_by_img, cls_to_cat_id, categories


class CocoDetectionDataset(Dataset):
    def __init__(self, coco_path: str | Path, images_dir: str | Path) -> None:
        self.coco = load_json(coco_path)
        self.images_dir = Path(images_dir)
        self.images, self.anns_by_img, self.cls_to_cat_id, self.categories = build_index(self.coco)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        img_id = int(img_info["id"])
        file_name = img_info["file_name"]
        img_path = self.images_dir / file_name
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        anns = self.anns_by_img.get(img_id, [])
        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        # cat_id_to_cls is inverse of cls_to_cat_id
        cat_id_to_cls = {v: k for k, v in self.cls_to_cat_id.items()}

        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            x1 = float(x)
            y1 = float(y)
            x2 = float(x + bw)
            y2 = float(y + bh)
            # clamp (defensive)
            x1 = max(0.0, min(x1, w - 1))
            y1 = max(0.0, min(y1, h - 1))
            x2 = max(0.0, min(x2, w))
            y2 = max(0.0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(int(cat_id_to_cls[int(ann["category_id"])]))
            areas.append(float(ann.get("area", bw * bh)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }

        # torchvision detection models expect tensors
        img_tensor = torch.from_numpy(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).numpy())
        # convert to CHW float32
        img_tensor = img_tensor.view(h, w, 3).permute(2, 0, 1).float() / 255.0
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def make_model(num_classes: int) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    train_ds = CocoDetectionDataset(args.train, args.images)
    val_ds = CocoDetectionDataset(args.val, args.images)

    num_classes = 1 + len(train_ds.categories)  # background + K
    model = make_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_loader))

        # quick val pass (loss only)
        model.train()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                val_loss += float(loss.item())
        val_loss = val_loss / max(1, len(val_loader))

        print(f"epoch={epoch} train_loss={avg_loss:.4f} val_loss={val_loss:.4f}")

    ckpt = {
        "model_state_dict": model.state_dict(),
        "categories": train_ds.categories,
        "cls_to_cat_id": train_ds.cls_to_cat_id,
    }
    torch.save(ckpt, out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()

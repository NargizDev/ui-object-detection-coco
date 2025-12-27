from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="Baseline hierarchical json")
    p.add_argument("--current", required=True, help="Current hierarchical json")
    p.add_argument("--out", required=True, help="Output diff json")
    return p.parse_args()


def bbox_to_tuple(bbox: List[float]) -> tuple:
    return tuple(round(x, 2) for x in bbox)


def pred_to_key(pred: Dict[str, Any]) -> tuple:
    return (pred['category_id'], bbox_to_tuple(pred['bbox']))


def compare_trees(baseline: List[Dict[str, Any]], current: List[Dict[str, Any]]) -> Dict[str, Any]:
    baseline_keys = {pred_to_key(p): p for p in baseline}
    current_keys = {pred_to_key(p): p for p in current}

    added = [p for k, p in current_keys.items() if k not in baseline_keys]
    removed = [p for k, p in baseline_keys.items() if k not in current_keys]
    changed = []

    for k in set(baseline_keys) & set(current_keys):
        b = baseline_keys[k]
        c = current_keys[k]
        # Recursive compare children
        child_diff = compare_trees(b.get('children', []), c.get('children', []))
        if child_diff['added'] or child_diff['removed'] or child_diff['changed']:
            changed.append({'baseline': b, 'current': c, 'child_diff': child_diff})

    return {
        'added': added,
        'removed': removed,
        'changed': changed
    }


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline)
    current_path = Path(args.current)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    baseline: Dict[str, List[Dict[str, Any]]] = json.loads(baseline_path.read_text(encoding="utf-8"))
    current: Dict[str, List[Dict[str, Any]]] = json.loads(current_path.read_text(encoding="utf-8"))

    diff = {}
    for img_key in set(baseline) | set(current):
        b_tree = baseline.get(img_key, [])
        c_tree = current.get(img_key, [])
        diff[img_key] = compare_trees(b_tree, c_tree)

    out_path.write_text(json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved diff: {out_path}")


if __name__ == "__main__":
    main()

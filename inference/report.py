from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any


def generate_report(diff_path: Path, out_path: Path) -> None:
    with open(diff_path, 'r', encoding='utf-8') as f:
        diff_data: Dict[str, Dict[str, List[Dict[str, Any]]]] = json.load(f)

    report = []
    total_screenshots = len(diff_data)
    total_elements = 0
    changed_screenshots = 0

    for img_name, changes in diff_data.items():
        added = len(changes.get('added', []))
        removed = len(changes.get('removed', []))
        changed = len(changes.get('changed', []))
        total_img_elements = added + removed + changed
        total_elements += total_img_elements

        if total_img_elements > 0:
            changed_screenshots += 1
            report.append({
                'Screenshot': img_name,
                'Added Elements': added,
                'Removed Elements': removed,
                'Changed Elements': changed,
                'Total Changes': total_img_elements
            })

    # Summary
    summary = f"""
# UI Change Detection Report

- **Total Screenshots Analyzed**: {total_screenshots}
- **Screenshots with Changes**: {changed_screenshots}
- **Total Elements Detected**: {total_elements}

## Detailed Changes

| Screenshot | Added Elements | Removed Elements | Changed Elements | Total Changes |
|------------|----------------|------------------|------------------|---------------|
"""

    for item in report:
        summary += f"| {item['Screenshot']} | {item['Added Elements']} | {item['Removed Elements']} | {item['Changed Elements']} | {item['Total Changes']} |\n"

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--diff", required=True, help="Path to diff.json")
    p.add_argument("--out", required=True, help="Output report file")
    args = p.parse_args()
    generate_report(Path(args.diff), Path(args.out))
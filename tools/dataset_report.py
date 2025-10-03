#!/usr/bin/env python3
# tools/dataset_report.py
"""
Generate dataset statistics report for YOLO format datasets.
Outputs image counts, class distribution, and average boxes per image.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def analyze_split(root, split):
    """
    Analyze a single dataset split.

    Args:
        root (Path): Dataset root directory
        split (str): Split name (e.g., 'train2017', 'val2017')

    Returns:
        dict: Statistics for the split
    """
    img_dir = root / f"images/{split}"
    lbl_dir = root / f"labels/{split}"

    if not img_dir.exists():
        return None

    imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    cls_counter = Counter()
    total_boxes = 0
    images_with_labels = 0

    for img in imgs:
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue

        images_with_labels += 1
        for line in lbl.read_text().strip().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:  # class_id x_center y_center width height
                continue
            try:
                cls = int(parts[0])
                cls_counter[cls] += 1
                total_boxes += 1
            except ValueError:
                continue

    return {
        "images": len(imgs),
        "images_with_labels": images_with_labels,
        "boxes": total_boxes,
        "avg_boxes_per_image": round(total_boxes / max(1, len(imgs)), 3),
        "avg_boxes_per_labeled_image": round(total_boxes / max(1, images_with_labels), 3),
        "num_classes": len(cls_counter),
        "class_histogram": dict(sorted(cls_counter.items())),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Dataset sanity report: image counts, class distribution, avg boxes/image."
    )
    ap.add_argument(
        "--root",
        required=True,
        help="Dataset root with images/labels like coco-mini/",
    )
    ap.add_argument("--out", default="reports/dataset_report.json", help="Output JSON path")
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["train2017", "val2017"],
        help="Dataset splits to analyze",
    )
    args = ap.parse_args()

    root = Path(args.root)
    report = {}

    print(f"Analyzing dataset: {root.resolve()}")
    print("=" * 60)

    for split in args.splits:
        stats = analyze_split(root, split)
        if stats is None:
            print(f"⚠ Split '{split}' not found, skipping...")
            continue

        report[split] = stats

        # Print summary
        print(f"\n{split.upper()}")
        print("-" * 60)
        print(f"  Images: {stats['images']}")
        print(f"  Images with labels: {stats['images_with_labels']}")
        print(f"  Total boxes: {stats['boxes']}")
        print(f"  Avg boxes/image: {stats['avg_boxes_per_image']:.3f}")
        print(f"  Avg boxes/labeled image: {stats['avg_boxes_per_labeled_image']:.3f}")
        print(f"  Unique classes: {stats['num_classes']}")

        # Show top 10 classes
        if stats['class_histogram']:
            top_classes = sorted(
                stats['class_histogram'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            print(f"  Top 10 classes:")
            for cls_id, count in top_classes:
                print(f"    Class {cls_id}: {count} boxes")

    # Save report
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print(f"✓ Report saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# tools/make_coco_subset.py
"""
Create a reproducible, small COCO subset for fast training/validation.
Max train images capped at 1000 per project requirements.
"""

import argparse
import random
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="Create a small COCO subset (≤1000 train images)."
    )
    ap.add_argument(
        "--coco",
        required=True,
        help="Root with images/{train2017,val2017} and labels/",
    )
    ap.add_argument("--out", default="coco-mini", help="Output directory")
    ap.add_argument("--max_train", type=int, default=1000, help="Max train images")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    random.seed(args.seed)
    root, out = Path(args.coco), Path(args.out)

    # Create output directories
    for p in [
        out / "images/train2017",
        out / "images/val2017",
        out / "labels/train2017",
        out / "labels/val2017",
    ]:
        p.mkdir(parents=True, exist_ok=True)

    # Process training split
    train_imgs = sorted((root / "images/train2017").glob("*.jpg"))
    print(f"Found {len(train_imgs)} train images")

    if len(train_imgs) > args.max_train:
        train_imgs = random.sample(train_imgs, args.max_train)
        print(f"Sampled {args.max_train} train images")

    def copy_split(img_paths, split):
        """Copy images and labels for a given split."""
        for img_p in img_paths:
            shutil.copy2(img_p, out / f"images/{split}" / img_p.name)
            lbl = root / f"labels/{split}" / (img_p.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, out / f"labels/{split}" / lbl.name)

    copy_split(train_imgs, "train2017")
    print(f"Copied {len(train_imgs)} train images and labels")

    # Process validation split (keep full or subset if desired)
    val_imgs = sorted((root / "images/val2017").glob("*.jpg"))
    print(f"Found {len(val_imgs)} val images")
    copy_split(val_imgs, "val2017")
    print(f"Copied {len(val_imgs)} val images and labels")

    # Create dataset YAML
    yaml_content = f"""# {out}/coco-mini.yaml
# Auto-generated COCO subset dataset configuration
path: {out.resolve()}
train: images/train2017
val: images/val2017

# COCO class names (80 classes)
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
"""
    (out / "coco-mini.yaml").write_text(yaml_content)
    print(f"✓ Wrote: {out / 'coco-mini.yaml'}")
    print(f"✓ Dataset ready with {len(train_imgs)} train images (max: {args.max_train})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# tools/tta_predict.py
"""
Test-Time Augmentation (TTA) for YOLO detection with ensemble box fusion.
Supports both Weighted Boxes Fusion (WBF) and Soft-NMS for merging predictions.
"""

import argparse
import glob
import os

import cv2
import numpy as np
from ensemble_boxes import soft_nms, weighted_boxes_fusion


def load_model(weights):
    """Load YOLO model from weights file."""
    try:
        from ultralytics import YOLO
        model = YOLO(weights)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure ultralytics is installed: pip install ultralytics")
        raise


def predict_single(model, image, imgsz, conf, iou):
    """
    Run inference on a single image.

    Args:
        model: YOLO model
        image: Input image (numpy array, RGB)
        imgsz: Image size for inference
        conf: Confidence threshold
        iou: IoU threshold for NMS

    Returns:
        tuple: (boxes, scores, labels) in normalized coordinates
    """
    results = model.predict(image, imgsz=imgsz, conf=conf, iou=iou, verbose=False)

    if len(results) == 0 or len(results[0].boxes) == 0:
        return [], [], []

    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy()  # xyxy format
    scores = r.boxes.conf.cpu().numpy()
    labels = r.boxes.cls.cpu().numpy().astype(int)

    # Normalize boxes to [0, 1]
    h, w = image.shape[:2]
    boxes_norm = boxes.copy()
    boxes_norm[:, [0, 2]] /= w
    boxes_norm[:, [1, 3]] /= h

    return boxes_norm.tolist(), scores.tolist(), labels.tolist()


def apply_tta(model, image, imgsz, conf, iou):
    """
    Apply test-time augmentation (horizontal, vertical, and both flips).

    Args:
        model: YOLO model
        image: Input image (numpy array, RGB)
        imgsz: Image size for inference
        conf: Confidence threshold
        iou: IoU threshold

    Returns:
        tuple: Lists of (boxes, scores, labels) for each augmentation
    """
    h, w = image.shape[:2]

    # Original + 3 flips
    augmentations = [
        ("original", image),
        ("hflip", image[:, ::-1]),
        ("vflip", image[::-1, :]),
        ("hvflip", image[::-1, ::-1]),
    ]

    all_boxes, all_scores, all_labels = [], [], []

    for aug_name, aug_img in augmentations:
        boxes, scores, labels = predict_single(model, aug_img, imgsz, conf, iou)

        # Un-flip boxes back to original orientation
        if "hflip" in aug_name and boxes:
            boxes = [[1 - x2, y1, 1 - x1, y2] for x1, y1, x2, y2 in boxes]
        if "vflip" in aug_name and boxes:
            boxes = [[x1, 1 - y2, x2, 1 - y1] for x1, y1, x2, y2 in boxes]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    return all_boxes, all_scores, all_labels


def fuse_boxes(boxes_list, scores_list, labels_list, method="wbf", iou_thr=0.55, skip_thr=0.0):
    """
    Fuse boxes from multiple predictions using WBF or Soft-NMS.

    Args:
        boxes_list: List of box predictions (normalized coordinates)
        scores_list: List of confidence scores
        labels_list: List of class labels
        method: Fusion method ("wbf" or "soft-nms")
        iou_thr: IoU threshold for fusion
        skip_thr: Skip boxes with confidence below this threshold

    Returns:
        tuple: Fused (boxes, scores, labels)
    """
    if method == "soft-nms":
        # Combine all predictions into single lists
        all_boxes, all_scores, all_labels = [], [], []
        for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
            all_boxes.extend(boxes)
            all_scores.extend(scores)
            all_labels.extend(labels)

        if not all_boxes:
            return [], [], []

        boxes, scores, labels = soft_nms(
            [all_boxes], [all_scores], [all_labels],
            sigma=0.5,
            iou_thr=iou_thr,
            thresh=skip_thr,
        )
        return boxes[0], scores[0], labels[0]
    else:  # WBF
        if not any(boxes_list):
            return [], [], []

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            iou_thr=iou_thr,
            skip_box_thr=skip_thr,
        )
        return boxes.tolist(), scores.tolist(), labels.tolist()


def draw_predictions(image, boxes, scores, labels, class_names=None):
    """
    Draw bounding boxes on image.

    Args:
        image: Input image (BGR format)
        boxes: List of boxes in pixel coordinates
        scores: List of confidence scores
        labels: List of class labels
        class_names: Optional list of class names

    Returns:
        Image with drawn boxes
    """
    img = image.copy()
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label_text = f"{int(label)}" if class_names is None else class_names[int(label)]
        text = f"{label_text}: {score:.2f}"
        cv2.putText(
            img, text, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    return img


def main():
    ap = argparse.ArgumentParser(
        description="TTA prediction with WBF/Soft-NMS fusion for YOLO models."
    )
    ap.add_argument("--weights", required=True, help="Path to model weights (.pt)")
    ap.add_argument("--source", required=True, help="Directory containing images")
    ap.add_argument("--out", default="tta_results", help="Output directory")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS")
    ap.add_argument(
        "--method",
        choices=["wbf", "soft-nms"],
        default="wbf",
        help="Box fusion method",
    )
    ap.add_argument("--fusion_iou", type=float, default=0.55, help="IoU threshold for fusion")
    args = ap.parse_args()

    # Create output directory
    os.makedirs(args.out, exist_ok=True)

    # Load model
    print(f"Loading model: {args.weights}")
    model = load_model(args.weights)

    # Process images
    image_files = glob.glob(os.path.join(args.source, "*.*"))
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    print(f"Found {len(image_files)} images in {args.source}")
    print(f"TTA method: {args.method.upper()}")

    for img_path in image_files:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Warning: Could not read {img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Apply TTA
        boxes_list, scores_list, labels_list = apply_tta(
            model, img_rgb, args.imgsz, args.conf, args.iou
        )

        # Fuse predictions
        boxes_fused, scores_fused, labels_fused = fuse_boxes(
            boxes_list, scores_list, labels_list,
            method=args.method,
            iou_thr=args.fusion_iou,
            skip_thr=args.conf,
        )

        # Denormalize boxes to pixel coordinates
        if boxes_fused:
            boxes_pixel = []
            for box in boxes_fused:
                x1, y1, x2, y2 = box
                boxes_pixel.append([x1 * w, y1 * h, x2 * w, y2 * h])

            # Draw and save
            img_out = draw_predictions(img_bgr, boxes_pixel, scores_fused, labels_fused)
            out_path = os.path.join(args.out, os.path.basename(img_path))
            cv2.imwrite(out_path, img_out)
            print(f"Saved: {out_path} ({len(boxes_fused)} detections)")
        else:
            print(f"No detections for {os.path.basename(img_path)}")

    print(f"\n✓ TTA predictions saved to: {args.out}")


if __name__ == "__main__":
    main()

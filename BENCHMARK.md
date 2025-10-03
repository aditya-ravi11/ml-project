# BENCHMARK

This document contains performance benchmarks for Hyper-YOLO model optimizations.

## Latency Benchmarks

ONNX model inference latency measured on CPU using ONNX Runtime.

| Model        | Format    | Avg Latency (ms) | Speedup |
|--------------|-----------|------------------|---------|
| Hyper-YOLO   | ONNX FP32 | TBD              | 1.0x    |
| Hyper-YOLO   | ONNX INT8 | TBD              | TBD     |

**Methodology:**
- **Hardware:** CPU only (ORT CPUExecutionProvider)
- **Image Size:** 640x640 (NCHW format)
- **Iterations:** 50 runs (after 5 warmup runs)
- **Input:** Random tensor data
- **Tool:** `tools/quantize_onnx.py`

### How to Reproduce

1. **Export model to ONNX:**
   ```bash
   # Follow README instructions to export trained model
   python ultralytics/utils/export_onnx.py --weights runs/train/weights/best.pt
   ```

2. **Quantize and benchmark:**
   ```bash
   python tools/quantize_onnx.py \
     --onnx model.onnx \
     --calib ./coco-mini/images/val2017 \
     --out model-int8.onnx \
     --imgsz 640
   ```

The script will automatically measure and compare FP32 vs INT8 latency.

## Test-Time Augmentation (TTA) Results

TTA improves detection robustness by ensembling predictions from multiple augmented views.

| Method        | Augmentations      | Fusion Method | mAP (TBD) |
|---------------|--------------------|---------------|-----------|
| Baseline      | None               | -             | TBD       |
| TTA + WBF     | H-flip, V-flip, HV | WBF           | TBD       |
| TTA + Soft-NMS| H-flip, V-flip, HV | Soft-NMS      | TBD       |

**Augmentations:**
- Original image
- Horizontal flip
- Vertical flip
- Horizontal + Vertical flip

**Tool:** `tools/tta_predict.py`

### How to Use TTA

```bash
python tools/tta_predict.py \
  --weights runs/train/weights/best.pt \
  --source ./examples/images \
  --out tta_results \
  --imgsz 640 \
  --conf 0.25 \
  --method wbf
```

## Dataset Statistics

Created using `tools/dataset_report.py` on COCO mini subset (≤1000 train images).

| Split       | Images | Boxes | Avg Boxes/Image | Classes |
|-------------|--------|-------|-----------------|---------|
| train2017   | TBD    | TBD   | TBD             | TBD     |
| val2017     | TBD    | TBD   | TBD             | TBD     |

**Constraint:** Training subset limited to maximum 1000 images per project requirements.

### Generate Dataset Report

```bash
python tools/dataset_report.py \
  --root ./coco-mini \
  --out reports/dataset_report.json
```

---

## Notes

- **TBD values** will be filled in after running the respective tools on your hardware
- All benchmarks use the default Hyper-YOLO model configuration
- INT8 quantization speedup varies by hardware (typically 2-4x on CPU)
- TTA increases inference time but can improve detection quality

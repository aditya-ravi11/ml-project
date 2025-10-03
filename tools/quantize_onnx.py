#!/usr/bin/env python3
# tools/quantize_onnx.py
"""
Static INT8 quantization for ONNX models with latency benchmarking.
Uses a small calibration set for quantization and measures CPU inference speed.
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantType,
    quantize_static,
)


class ImageFolderCalib(CalibrationDataReader):
    """Calibration data reader for ONNX Runtime quantization."""

    def __init__(self, folder, input_name="images", size=640, limit=300):
        """
        Initialize calibration data reader.

        Args:
            folder (str): Path to folder containing calibration images
            input_name (str): Name of the model input tensor
            size (int): Image size for resizing
            limit (int): Maximum number of calibration images to use
        """
        self.input_name = input_name
        self.size = size
        self.paths = glob.glob(os.path.join(folder, "**", "*.jpg"), recursive=True)[
            :limit
        ]
        self.iter = None
        print(f"Loaded {len(self.paths)} calibration images from {folder}")

    def get_next(self):
        """Get next calibration batch."""
        if self.iter is None:

            def gen():
                for p in self.paths:
                    im = cv2.imread(p)
                    if im is None:
                        continue
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im, (self.size, self.size)).astype(np.float32) / 255.0
                    im = np.transpose(im, (2, 0, 1))[None]  # NCHW
                    yield {self.input_name: im}

            self.iter = gen()
        return next(self.iter, None)


def bench(session, input_name="images", imgsz=640, iters=50):
    """
    Benchmark ONNX model inference latency.

    Args:
        session: ONNX Runtime inference session
        input_name (str): Name of input tensor
        imgsz (int): Input image size
        iters (int): Number of iterations for benchmarking

    Returns:
        float: Average latency in milliseconds
    """
    x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    # Warmup
    for _ in range(5):
        session.run(None, {input_name: x})

    # Benchmark
    t0 = time.time()
    for _ in range(iters):
        session.run(None, {input_name: x})
    return (time.time() - t0) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser(
        description="Static INT8 quantization for ONNX + latency bench."
    )
    ap.add_argument("--onnx", required=True, help="Path to input ONNX model")
    ap.add_argument("--out", default="model-int8.onnx", help="Output quantized model path")
    ap.add_argument(
        "--calib",
        required=True,
        help="Folder of calibration images (e.g., coco/images/val2017)",
    )
    ap.add_argument("--input_name", default="images", help="Model input tensor name")
    ap.add_argument("--imgsz", type=int, default=640, help="Input image size")
    ap.add_argument("--calib_limit", type=int, default=300, help="Max calibration images")
    args = ap.parse_args()

    print(f"Quantizing {args.onnx} to INT8...")
    print(f"Calibration: {args.calib}")

    # Create calibration data reader
    dr = ImageFolderCalib(args.calib, args.input_name, args.imgsz, args.calib_limit)

    # Quantize model
    quantize_static(
        args.onnx,
        args.out,
        dr,
        weight_type=QuantType.QInt8,
    )
    print(f"✓ Saved quantized model: {args.out}")

    # Benchmark both models
    print("\nBenchmarking inference latency (CPU)...")
    s_fp32 = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    s_int8 = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])

    ms_fp32 = bench(s_fp32, args.input_name, args.imgsz)
    ms_int8 = bench(s_int8, args.input_name, args.imgsz)

    print(f"\n{'='*50}")
    print(f"Latency (avg over 50 runs, CPU)")
    print(f"{'='*50}")
    print(f"FP32: {ms_fp32:.2f} ms")
    print(f"INT8: {ms_int8:.2f} ms")
    print(f"Speedup: {ms_fp32/ms_int8:.2f}x")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

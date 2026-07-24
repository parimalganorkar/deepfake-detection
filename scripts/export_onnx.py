"""Export the ResNet-50 classifier to ONNX, quantize it, and benchmark CPU inference.

Usage:
    python -m scripts.export_onnx            # export + quantize + benchmark
    python -m scripts.export_onnx --no-bench # export + quantize only

Outputs (into settings.onnx_dir, which is gitignored):
    resnet50.onnx         fp32 ONNX export
    resnet50.quant.onnx   dynamically quantized (int8) ONNX export

Then set DEEPFAKE_USE_ONNX=1 (env) to make the app/API use the ONNX path.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from core.config import settings
from core.logging_config import get_logger
from core.models import load_classifier

logger = get_logger("export_onnx")

OPSET = 17


def export_fp32(onnx_path: Path) -> None:
    model, class_to_idx, _ = load_classifier()
    model.eval()
    dummy = torch.randn(1, 3, settings.img_size, settings.img_size)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
    )
    logger.info("Exported fp32 ONNX -> %s (class_to_idx=%s)", onnx_path, class_to_idx)


def quantize(fp32_path: Path, quant_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(quant_path),
        weight_type=QuantType.QInt8,
    )
    logger.info("Quantized ONNX (int8) -> %s", quant_path)


def _mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _bench_torch(n: int, batch: int) -> float:
    model, _, _ = load_classifier()
    x = torch.randn(batch, 3, settings.img_size, settings.img_size)
    with torch.no_grad():
        model(x)  # warmup
        start = time.perf_counter()
        for _ in range(n):
            model(x)
        return (time.perf_counter() - start) / n * 1000.0


def _bench_onnx(path: Path, n: int, batch: int) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.randn(batch, 3, settings.img_size, settings.img_size).astype(np.float32)
    sess.run(None, {name: x})  # warmup
    start = time.perf_counter()
    for _ in range(n):
        sess.run(None, {name: x})
    return (time.perf_counter() - start) / n * 1000.0


def benchmark(fp32_path: Path, quant_path: Path, n: int = 30, batch: int = 1) -> None:
    logger.info("Benchmarking (%d iters, batch=%d)...", n, batch)
    t_torch = _bench_torch(n, batch)
    t_onnx = _bench_onnx(fp32_path, n, batch)
    t_quant = _bench_onnx(quant_path, n, batch)

    print("\n================ CPU inference benchmark ================")
    print(f"{'backend':<22}{'latency/img (ms)':>18}{'model size (MB)':>18}{'speedup':>10}")
    print("-" * 68)
    print(f"{'PyTorch (fp32)':<22}{t_torch:>18.1f}{'—':>18}{'1.00x':>10}")
    print(f"{'ONNX Runtime (fp32)':<22}{t_onnx:>18.1f}{_mb(fp32_path):>18.1f}{t_torch / t_onnx:>9.2f}x")
    print(f"{'ONNX Runtime (int8)':<22}{t_quant:>18.1f}{_mb(quant_path):>18.1f}{t_torch / t_quant:>9.2f}x")
    print("========================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export + quantize + benchmark the ResNet-50 classifier.")
    parser.add_argument("--no-bench", action="store_true", help="skip the benchmark")
    parser.add_argument("--iters", type=int, default=30, help="benchmark iterations")
    args = parser.parse_args()

    fp32_path = Path(settings.onnx_model_path)
    quant_path = Path(settings.onnx_quant_path)

    export_fp32(fp32_path)
    quantize(fp32_path, quant_path)

    if not args.no_bench:
        benchmark(fp32_path, quant_path, n=args.iters)


if __name__ == "__main__":
    main()

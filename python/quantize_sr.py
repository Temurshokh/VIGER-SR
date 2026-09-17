"""Quantize a trained VIGER SR ONNX model to a smaller dynamic-int8 model."""

from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic INT8 quantization for VIGER SR")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(args.input),
        model_output=str(args.output),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    print(f"input={args.input} size={args.input.stat().st_size:,} bytes")
    print(f"output={args.output} size={args.output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()

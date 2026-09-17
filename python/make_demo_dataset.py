"""Generate a small synthetic SR dataset for an end-to-end smoke test.

The images are deliberately synthetic: gradients, circles, lines, grids and
controlled noise make it useful for validating the pipeline, not for measuring
photorealistic SR quality.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_image(seed: int, size: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]

    base = np.zeros((size, size, 3), dtype=np.float32)
    base[..., 0] = 30 + 180 * (xx / max(1, size - 1))
    base[..., 1] = 25 + 190 * (yy / max(1, size - 1))
    base[..., 2] = 55 + 90 * (0.5 + 0.5 * np.sin(xx / 21.0 + seed * 0.3))

    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)

    for _ in range(12):
        x = int(rng.integers(0, size))
        y = int(rng.integers(0, size))
        radius = int(rng.integers(size // 30, size // 8 + 1))
        color = tuple(int(v) for v in rng.integers(40, 245, size=3))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=max(1, size // 160))

    for y in range(0, size, max(16, size // 8)):
        draw.line((0, y, size - 1, y), fill=(240, 240, 240), width=1)
    for x in range(0, size, max(16, size // 8)):
        draw.line((x, 0, x, size - 1), fill=(230, 230, 230), width=1)

    for _ in range(8):
        x1, y1 = rng.integers(0, size, size=2)
        x2, y2 = rng.integers(0, size, size=2)
        color = tuple(int(v) for v in rng.integers(10, 255, size=3))
        draw.line((int(x1), int(y1), int(x2), int(y2)), fill=color, width=max(1, size // 120))

    array = np.asarray(image, dtype=np.float32)
    array += rng.normal(0.0, 1.5, array.shape).astype(np.float32)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/demo"))
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    if args.count <= 0 or args.size < 32:
        raise ValueError("count must be positive and size must be at least 32")

    args.output.mkdir(parents=True, exist_ok=True)
    for index in range(args.count):
        make_image(index, args.size).save(args.output / f"demo_{index:03d}.png")

    print(f"generated={args.count} images")
    print(f"folder={args.output}")


if __name__ == "__main__":
    main()

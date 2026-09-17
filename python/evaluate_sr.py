"""Evaluate a VIGER SR output against a matching reference image.

Reports image dimensions, file sizes, MSE, PSNR and a luminance-window SSIM
implementation with no dependency on scikit-image.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(np.mean((reference - candidate) ** 2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(1.0 / mse)


def box_filter(image: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return image
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="reflect")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    ) / float(size * size)


def ssim(reference: np.ndarray, candidate: np.ndarray, radius: int = 3) -> float:
    ref = 0.2126 * reference[..., 0] + 0.7152 * reference[..., 1] + 0.0722 * reference[..., 2]
    cand = 0.2126 * candidate[..., 0] + 0.7152 * candidate[..., 1] + 0.0722 * candidate[..., 2]

    mu_ref = box_filter(ref, radius)
    mu_cand = box_filter(cand, radius)
    mu_ref2 = box_filter(ref * ref, radius)
    mu_cand2 = box_filter(cand * cand, radius)
    mu_cross = box_filter(ref * cand, radius)

    sigma_ref = np.maximum(0.0, mu_ref2 - mu_ref * mu_ref)
    sigma_cand = np.maximum(0.0, mu_cand2 - mu_cand * mu_cand)
    sigma_cross = mu_cross - mu_ref * mu_cand

    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_ref * mu_cand + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_ref * mu_ref + mu_cand * mu_cand + c1) * (sigma_ref + sigma_cand + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path, help="ground-truth/reference image")
    parser.add_argument("candidate", type=Path, help="VIGER output image")
    args = parser.parse_args()

    reference = load_rgb(args.reference)
    candidate = load_rgb(args.candidate)
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: reference={reference.shape}, candidate={candidate.shape}")

    score_psnr = psnr(reference, candidate)
    score_ssim = ssim(reference, candidate)
    print(f"reference={args.reference} size={args.reference.stat().st_size:,} bytes")
    print(f"candidate={args.candidate} size={args.candidate.stat().st_size:,} bytes")
    print(f"shape={candidate.shape[1]}x{candidate.shape[0]}")
    print(f"PSNR={score_psnr:.4f} dB")
    print(f"SSIM={score_ssim:.6f}")


if __name__ == "__main__":
    main()

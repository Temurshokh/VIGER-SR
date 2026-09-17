"""Train a compact residual super-resolution refinement model.

The model receives a bicubic-upscaled image and learns a residual correction at
that same resolution. Python is training-only; inference is exported to ONNX
for the native C++ runtime.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset


class ImageFolderPairs(Dataset):
    def __init__(self, root: Path, patch_size: int = 96, scale: int = 2) -> None:
        self.paths = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        )
        self.patch_size = patch_size
        self.scale = scale
        if patch_size % scale != 0:
            raise ValueError("patch_size must be divisible by scale")
        if not self.paths:
            raise ValueError(f"No images found in {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB")
        width, height = image.size

        if min(width, height) < self.patch_size:
            scale_up = self.patch_size / min(width, height)
            image = image.resize(
                (max(self.patch_size, round(width * scale_up)), max(self.patch_size, round(height * scale_up))),
                Image.Resampling.LANCZOS,
            )
            width, height = image.size

        left = random.randint(0, width - self.patch_size)
        top = random.randint(0, height - self.patch_size)
        image = image.crop((left, top, left + self.patch_size, top + self.patch_size))

        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.25:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if random.random() < 0.25:
            image = image.rotate(90, expand=False)

        hr = np.asarray(image, dtype=np.float32) / 255.0

        # Simulate a small amount of capture/compression degradation before the
        # spatial reduction. The final bicubic resize matches the native C++ path.
        degraded = image
        if random.random() < 0.45:
            degraded = degraded.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.8)))
        small = degraded.resize(
            (self.patch_size // self.scale, self.patch_size // self.scale),
            Image.Resampling.BICUBIC,
        )
        upscaled = small.resize((self.patch_size, self.patch_size), Image.Resampling.BICUBIC)
        lr = np.asarray(upscaled, dtype=np.float32) / 255.0
        lr += np.random.normal(0.0, 0.004, lr.shape).astype(np.float32)
        lr = np.clip(lr, 0.0, 1.0)

        return (
            torch.from_numpy(lr).permute(2, 0, 1),
            torch.from_numpy(hr).permute(2, 0, 1),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.2 * self.body(x)


class TinyResidualSR(nn.Module):
    def __init__(self, channels: int = 48, blocks: int = 8) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.head(x)
        features = self.body(features)
        residual = self.tail(features)
        return torch.clamp(x + 0.15 * residual, 0.0, 1.0)


def charbonnier(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = prediction - target
    return torch.sqrt(diff * diff + 1e-6).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tiny_sr.onnx"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--scale", type=int, default=2, choices=(2, 4))
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    dataset = ImageFolderPairs(args.data, patch_size=args.patch_size, scale=args.scale)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyResidualSR(channels=args.channels, blocks=args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for lr_image, hr_image in loader:
            lr_image = lr_image.to(device)
            hr_image = hr_image.to(device)
            prediction = model(lr_image)
            loss = charbonnier(prediction, hr_image)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()

        average = running / max(1, len(loader))
        print(f"epoch={epoch + 1}/{args.epochs} loss={average:.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 3, args.patch_size, args.patch_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["image"],
        output_names=["enhanced"],
        dynamic_axes={"image": {2: "height", 3: "width"}, "enhanced": {2: "height", 3: "width"}},
        opset_version=17,
        dynamo=False,
    )

    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"exported={args.output}")
    print(f"parameters={parameter_count:,}")


if __name__ == "__main__":
    main()

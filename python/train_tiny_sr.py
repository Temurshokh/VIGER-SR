"""Train a compact residual super-resolution model.

This is intentionally small and framework-light at the model level.
The script uses PyTorch for training and exports an ONNX model for the
native C++ runtime in a later milestone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class ImageFolderPairs(Dataset):
    def __init__(self, root: Path, patch_size: int = 96, scale: int = 2) -> None:
        self.paths = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        self.patch_size = patch_size
        self.scale = scale
        if not self.paths:
            raise ValueError(f"No images found in {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB")
        width, height = image.size
        size = min(width, height)
        if size < self.patch_size:
            image = image.resize((self.patch_size, self.patch_size), Image.Resampling.LANCZOS)
        else:
            left = (width - self.patch_size) // 2
            top = (height - self.patch_size) // 2
            image = image.crop((left, top, left + self.patch_size, top + self.patch_size))

        hr = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0
        small = image.resize(
            (self.patch_size // self.scale, self.patch_size // self.scale),
            Image.Resampling.BICUBIC,
        )
        small = small.resize((self.patch_size, self.patch_size), Image.Resampling.BICUBIC)
        lr = torch.from_numpy(__import__("numpy").array(small)).permute(2, 0, 1).float() / 255.0
        return lr, hr


class TinyResidualSR(nn.Module):
    def __init__(self, channels: int = 24, blocks: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(blocks):
            layers.extend(
                [
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, 3, padding=1),
                ]
            )
        layers.append(nn.Conv2d(channels, 3, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.body(x)
        return torch.clamp(x + residual, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tiny_sr.onnx"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--scale", type=int, default=2, choices=(2, 4))
    args = parser.parse_args()

    dataset = ImageFolderPairs(args.data, scale=args.scale)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyResidualSR().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for lr_image, hr_image in loader:
            lr_image = lr_image.to(device)
            hr_image = hr_image.to(device)
            prediction = model(lr_image)
            loss = criterion(prediction, hr_image)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"epoch={epoch + 1}/{args.epochs} loss={running / len(loader):.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 3, 96, 96, device=device)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["image"],
        output_names=["enhanced"],
        dynamic_axes={"image": {2: "height", 3: "width"}, "enhanced": {2: "height", 3: "width"}},
        opset_version=17,
    )
    print(f"exported={args.output}")
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()

"""Local Python super-resolution engine.

For the Telegram experiment we use pretrained Swin2SR weights locally. There is
no server and no native executable. The model files are downloaded once by the
Transformers cache and then reused locally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

MODELS = {
    2: "caidas/swin2SR-classical-sr-x2-64",
    4: "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
}


class SREngine:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models: dict[int, tuple[object, object]] = {}

    def _load(self, scale: int):
        if scale not in MODELS:
            raise ValueError("scale must be 2 or 4")
        if scale not in self._models:
            name = MODELS[scale]
            print(f"[VIGER SR] loading {name} on {self.device}...")
            processor = AutoImageProcessor.from_pretrained(name)
            model = Swin2SRForImageSuperResolution.from_pretrained(name).to(self.device)
            model.eval()
            self._models[scale] = (processor, model)
        return self._models[scale]

    @torch.inference_mode()
    def enhance(self, image: Image.Image, scale: int) -> Image.Image:
        processor, model = self._load(scale)
        rgb = image.convert("RGB")
        inputs = processor(rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = model(**inputs)
        output = outputs.reconstruction.squeeze(0).float().cpu().clamp_(0, 1).numpy()
        output = np.moveaxis(output, 0, -1)
        output = (output * 255.0).round().astype(np.uint8)
        return Image.fromarray(output)


def save_4k_or_less(image: Image.Image, path: Path) -> None:
    """Avoid creating unnecessarily huge Telegram files while keeping 4K output."""
    max_w, max_h = 3840, 2160
    if image.width > max_w or image.height > max_h:
        ratio = min(max_w / image.width, max_h / image.height)
        image = image.resize((round(image.width * ratio), round(image.height * ratio)), Image.Resampling.LANCZOS)
    image.save(path, format="PNG", optimize=True)

"""Local Python super-resolution engine.

The current Telegram experiment uses pretrained Swin2SR weights, but inference is
performed locally. We intentionally implement the small Swin2SR preprocessing
step ourselves so the project does not need torchvision for image preprocessing.
The model weights are downloaded once from Hugging Face and then cached locally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Swin2SRForImageSuperResolution

ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE = ROOT / ".cache" / "huggingface"

MODELS = {
    2: "caidas/swin2SR-classical-sr-x2-64",
    4: "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
}


def _prepare_input(image: Image.Image) -> tuple[torch.Tensor, int, int]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    array = np.asarray(rgb, dtype=np.float32) / 255.0

    # Swin2SR expects H/W divisible by 8. Symmetric padding matches the
    # reference processor's behavior without pulling in torchvision.
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    if pad_h or pad_w:
        array = np.pad(
            array,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="symmetric",
        )

    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor, width, height


class SREngine:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models: dict[int, Swin2SRForImageSuperResolution] = {}
        self._failures: dict[int, str] = {}

    def _load(self, scale: int) -> Swin2SRForImageSuperResolution:
        if scale not in MODELS:
            raise ValueError("scale must be 2 or 4")
        if scale in self._models:
            return self._models[scale]
        if scale in self._failures:
            raise RuntimeError(self._failures[scale])

        name = MODELS[scale]
        cache_dir = str(MODEL_CACHE)
        print(f"[VIGER SR] loading {name} on {self.device}...")
        try:
            model = Swin2SRForImageSuperResolution.from_pretrained(
                name,
                cache_dir=cache_dir,
            ).to(self.device)
            model.eval()
            self._models[scale] = model
            return model
        except Exception as exc:
            message = (
                f"Could not load Swin2SR '{name}'.\n"
                "The first run needs internet access to download the pretrained weights.\n"
                f"Original error: {type(exc).__name__}: {exc}"
            )
            self._failures[scale] = message
            raise RuntimeError(message) from exc

    @torch.inference_mode()
    def enhance(self, image: Image.Image, scale: int) -> Image.Image:
        model = self._load(scale)
        pixel_values, original_width, original_height = _prepare_input(image)
        pixel_values = pixel_values.to(self.device)
        outputs = model(pixel_values=pixel_values)

        output = outputs.reconstruction.squeeze(0).float().cpu().clamp_(0, 1).numpy()
        output = np.moveaxis(output, 0, -1)
        output = (output * 255.0).round().astype(np.uint8)

        # Remove the image-space padding after the model's upscale.
        output = output[: original_height * scale, : original_width * scale, :]
        return Image.fromarray(output, mode="RGB")


def save_4k_or_less(image: Image.Image, path: Path) -> None:
    """Cap output at UHD dimensions so Telegram files stay reasonable."""
    max_w, max_h = 3840, 2160
    if image.width > max_w or image.height > max_h:
        ratio = min(max_w / image.width, max_h / image.height)
        image = image.resize(
            (round(image.width * ratio), round(image.height * ratio)),
            Image.Resampling.LANCZOS,
        )
    image.save(path, format="PNG", optimize=True)

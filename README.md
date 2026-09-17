# VIGER SR

**Small model. Big pixels.**

VIGER SR is a native C++ super-resolution experiment with a real learned-model path, plus a small local chat runtime foundation.

## What connects to what

```text
                         TRAINING TIME

   high-quality images ──> Python / PyTorch
                              │
                              ├── TinyResidualSR
                              ├── validation / loss
                              └── export -> model.onnx
                                           │
                                           v
                                    model.onnx / INT8

                         RUN TIME

   image.ppm ─> C++ decoder ─> bicubic upscale ─> ONNX Runtime ─> image.ppm
                                      │                 │
                                      └── baseline ─────┘

   user ─> viger-chat ─> ChatSession ─> IChatModel
                                      │
                                      ├── fallback (built in)
                                      └── future local neural provider
```

Python is used for **training only**. The image inference runtime is C++.

## Current targets

- `viger-sr` — dependency-light C++ baseline.
- `viger-sr-neural` — native ONNX Runtime inference target (enabled when ONNX Runtime is installed).
- `viger-chat` — local conversation/session layer with a deterministic fallback provider.

## SR pipeline

The learned model is deliberately a **same-resolution residual refiner**. The native runtime first enlarges the input and then asks the model to predict a better reconstruction. This keeps the model simple and makes its role measurable.

```text
input
  -> bicubic upscale
  -> neural residual refinement
  -> clamping / output
```

The baseline path is still useful as a control because it lets us compare learned reconstruction against a non-neural result.

## Build the C++ core

```bash
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build --output-on-failure -C Release
```

Run the baseline:

```bash
# Windows
build\Release\viger-sr.exe input.ppm output.ppm 2 0.20

# Linux/macOS
./build/viger-sr input.ppm output.ppm 2 0.20
```

The first runtime milestone intentionally uses binary PPM, so the core stays free of image-codec dependencies. PNG/JPEG support is a separate I/O milestone.

## Train the SR model

Create a folder containing training images, then:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r python/requirements.txt
python python/train_tiny_sr.py --data ./data/train --epochs 20 --output artifacts/tiny_sr.onnx
```

The training script builds degraded inputs, trains a residual network, and exports a dynamic-size ONNX graph.

Quantize it:

```bash
python python/quantize_sr.py artifacts/tiny_sr.onnx artifacts/tiny_sr.int8.onnx
```

The quantizer prints the exact before/after model sizes instead of assuming a compression ratio.

## Enable native neural inference

Install ONNX Runtime for your platform and point CMake at the installation root:

```bash
cmake -S . -B build-neural \
  -DVIGER_SR_ENABLE_ONNX=ON \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build-neural --config Release
```

Then:

```bash
build/Release/viger-sr-neural input.ppm output.ppm artifacts/tiny_sr.onnx 2
```

The model is executed locally by ONNX Runtime; there is no VIGER backend service involved.

## Chat foundation

`viger-chat` is intentionally an interface-first design. The session owns conversation history and calls an `IChatModel` provider. The built-in provider is deterministic and exists only to test the plumbing.

```text
ChatSession
     |
     v
 IChatModel
  /      \
Fallback  local neural provider
```

This means a small local language model can be added without rewriting the session, commands, or future UI.

Run it:

```bash
build\Release\viger-chat.exe
```

Then try:

```text
hello
/history
/clear
/quit
```

## Tiny local language-model sandbox

`python/train_tiny_chat.py` contains a deliberately small byte-level causal Transformer for experimentation. It is not presented as a general assistant; its job is to provide a concrete local-model target for the chat interface.

Example:

```bash
python python/train_tiny_chat.py --data ./data/chat.txt --steps 3000 --output artifacts/tiny_chat.pt
```

## Roadmap

1. Native PNG/JPEG input/output.
2. Train a stronger SR model and publish benchmark images.
3. Add tiled inference for very large photos.
4. Add FP16 / INT8 model variants and automatic model-size reports.
5. Add PSNR/SSIM and perceptual benchmark tooling.
6. Add a concrete local neural `IChatModel` provider.
7. Add a desktop UI only after the native core is stable.

## Design rule

**Use AI where a trained model actually improves reconstruction. Keep the runtime native. Measure the result.**

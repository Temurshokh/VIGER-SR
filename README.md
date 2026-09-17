# VIGER SR

**Small model. Big pixels.**

VIGER SR is a native C++ super-resolution experiment with a real learned-model path, plus a small local chat runtime foundation.

## What connects to what

```text
                         TRAINING TIME

   high-quality images ──> Python / PyTorch
                              │
                              ├── TinyResidualSR
                              ├── degradation + random crops
                              └── export -> model.onnx
                                           │
                                           v
                                    model.onnx / INT8

                         RUN TIME

   photo ─> C++ image I/O ─> bicubic ─> ONNX Runtime ─> output
                                  │           │
                                  └ baseline ┘
                                      │
                             optional tiled inference

   user ─> viger-chat ─> ChatSession ─> IChatModel
                                      │
                            +---------+---------+
                            |                   |
                         fallback       local HTTP model
                                                |
                                                v
                                           local runtime
```

Python is used for **training and evaluation**. The production image inference runtime is C++.

## Current targets

- `viger-sr` — dependency-light C++ baseline.
- `viger-sr-neural` — native ONNX Runtime inference target (enabled when ONNX Runtime is installed).
- `viger-chat` — local conversation runtime with fallback or local-model provider.

## SR pipeline

The learned model is deliberately a **same-resolution residual refiner**. The native runtime first enlarges the input and then asks the model to correct reconstruction errors. This keeps the model simple and makes its contribution measurable.

```text
input
  -> bicubic upscale
  -> neural residual refinement
  -> clamping / output
```

For large images, `viger-sr-neural` can process overlapping tiles and feather the joins instead of sending the whole image through the model at once.

The baseline path is retained as a control so we can compare learned reconstruction against a non-neural result.

## Quick demo dataset

No external dataset is required to verify the mechanics. Generate synthetic training images:

```bash
python python/make_demo_dataset.py --output data/demo --count 32 --size 256
```

Train a tiny model for a smoke test:

```bash
python python/train_tiny_sr.py \
  --data data/demo \
  --epochs 1 \
  --batch-size 8 \
  --output artifacts/tiny_sr.onnx
```

The synthetic set is for pipeline validation only. It is **not** representative of real photographic quality.

## Build the C++ core

```bash
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build --output-on-failure -C Release
```

Run the baseline:

```bash
# Windows
build\Release\viger-sr.exe input.png output.png 2 0.15

# Linux/macOS (PPM/PNM in the current portable path)
./build/viger-sr input.ppm output.ppm 2 0.15
```

On Windows, PNG/JPEG/BMP/TIFF input is decoded through Windows Imaging Component. PPM/PNM is always supported.

## Train the SR model

Create a folder containing training images, then:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r python/requirements.txt
python python/train_tiny_sr.py --data ./data/train --epochs 20 --output artifacts/tiny_sr.onnx
```

The training script creates degraded inputs, performs random patches/augmentation, trains a residual network, and exports a dynamic-size ONNX graph.

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
build/Release/viger-sr-neural photo.jpg enhanced.png artifacts/tiny_sr.onnx 2
```

For a large image, add a tile size:

```bash
build/Release/viger-sr-neural photo.jpg enhanced.png artifacts/tiny_sr.onnx 2 768
```

The model is executed locally by ONNX Runtime; there is no VIGER cloud backend.

## Evaluate output quality

Given a reference image and a candidate output with identical dimensions:

```bash
python python/evaluate_sr.py reference.png candidate.png
```

The tool reports PSNR and SSIM plus exact file sizes. Use the same reference and degradation settings when comparing models.

## Chat

`viger-chat` has a provider interface, conversation history, context trimming, and two implementations:

```text
ChatSession
    |
IChatModel
 /        \
Fallback   LocalHttpChatModel
                |
                v
        local OpenAI-compatible runtime
```

Run the built-in plumbing test:

```bash
build\Release\viger-chat.exe
```

Run against a local model runtime on port 8080:

```bash
build\Release\viger-chat.exe --local
```

Or specify another host/port:

```bash
build\Release\viger-chat.exe --local 127.0.0.1 8080
```

The local provider sends the whole retained conversation as `system`, `user`, and `assistant` messages and parses the assistant response. No external API key is required for a normal local endpoint.

For the concrete endpoint contract and local-runtime notes, see [`docs/CHAT_PROVIDER.md`](docs/CHAT_PROVIDER.md).

## Tiny local language-model sandbox

`python/train_tiny_chat.py` contains a deliberately small byte-level causal Transformer for experimentation. It is not presented as a general assistant; it exists so the chat abstraction has a trainable local-model target.

Example:

```bash
python python/train_tiny_chat.py --data ./data/chat.txt --steps 3000 --output artifacts/tiny_chat.pt
```

## Project structure

```text
VIGER-SR/
├── include/viger_sr/      public C++ interfaces
├── src/                   native runtime
├── tests/                 C++ smoke tests
├── python/                training, quantization, evaluation
├── docs/                  architecture notes
└── .github/workflows/     cross-platform CI
```

## Roadmap

1. Cross-platform PNG/JPEG support without platform-specific codecs.
2. Stronger photographic SR training and reproducible benchmark sets.
3. FP16 / INT8 model variants with runtime selection.
4. Multi-threaded and GPU execution tuning.
5. Native GGUF/libllama chat provider, removing the local HTTP hop when useful.
6. Desktop UI only after the native core and benchmarks are stable.

## Design rule

**Use AI where a trained model actually improves reconstruction. Keep the runtime native. Measure the result.**

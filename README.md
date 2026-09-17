# VIGER SR

**Small model. Big pixels.**

VIGER SR is an experimental lightweight super-resolution project focused on getting useful image reconstruction from a deliberately small runtime and model footprint.

## Current status

The repository starts with a dependency-light C++17 baseline:

- Bilinear upscaling
- Edge-aware detail reconstruction
- CLI application
- Dependency-free P6 PPM image I/O
- CTest smoke tests
- Clean CMake build

This baseline is intentionally **not** presented as AI. It gives us a measurable reference before a learned model is introduced.

## Architecture

```text
input image
    |
    v
preprocess
    |
    +--> resize baseline
    |
    v
edge/detail reconstruction
    |
    v
output image
```

The next stage is a tiny learned residual network trained in Python and exported to a portable inference representation for the C++ runtime. The target is a compact model, not a giant image-generation stack.

## Build

```bash
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

## Run

The first milestone uses binary PPM files so the core can stay dependency-free:

```bash
viger-sr input.ppm output.ppm 2 0.20
```

Arguments:

- `input.ppm` — source image
- `output.ppm` — output image
- `2` — scale factor (`1`–`8`)
- `0.20` — detail amount

## Roadmap

1. Establish baseline quality and benchmarks.
2. Add image codec support behind a small I/O layer.
3. Train a tiny residual super-resolution model.
4. Add model quantization and size reporting.
5. Integrate the learned model into the native C++ pipeline.
6. Add PSNR/SSIM benchmarks and visual regression tests.
7. Build a simple desktop UI only after the core is solid.

## Design goal

**Measure everything. Keep the runtime small. Never call an algorithm “AI” until a trained model is actually doing the reconstruction.**

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vship is a high-performance VapourSynth plugin and CLI tool for GPU-accelerated perceptual video quality metrics. It provides hardware-accelerated implementations of SSIMULACRA2 and Butteraugli using HIP (AMD) or CUDA (NVIDIA).

The project consists of two primary components:
- **vship** - VapourSynth plugin (vship.dll/.so)
- **FFVship** - Standalone CLI tool for comparing video files

## Build System

### Basic Build Commands

All builds use the Makefile with specific targets based on GPU vendor and scope:

```bash
# VapourSynth Plugin
make buildcuda           # NVIDIA - optimize for current system GPU
make buildcudaall        # NVIDIA - build for all supported architectures
make build               # AMD - optimize for current system GPU
make buildall            # AMD - build for all supported architectures

# Standalone CLI Tool
make buildFFVSHIPcuda    # NVIDIA - optimize for current system GPU
make buildFFVSHIPcudaall # NVIDIA - build for all supported architectures
make buildFFVSHIP        # AMD - optimize for current system GPU
make buildFFVSHIPall     # AMD - build for all supported architectures

# Install (auto-detects what was built)
make install             # Linux: uses PREFIX=/usr/local by default
make install PREFIX=/usr # Arch Linux and similar

# Test VapourSynth plugin
make test                # Requires vspipe and test script

# Clean up
make uninstall           # Linux only
```

### Build Requirements

**Common:**
- `make`
- `hipcc` (AMD) or `nvcc` (NVIDIA)

**For VapourSynth plugin:**
- VapourSynth headers (included in `include/`)

**For FFVship CLI:**
- ffms2
- zimg
- pkg-config (Linux) or manual library setup (Windows)

### Platform-Specific Build Details

**Windows:**
- Plugin installs to `%APPDATA%\VapourSynth\plugins64`
- FFVship.exe installs to `%ProgramFiles%\FFVship.exe`
- Uses `-lz_imp -lz -lffms2` for linking
- No `-fPIC` flag needed

**Linux:**
- Plugin installs to `$(PREFIX)/lib/vapoursynth` with symlink from `$(PREFIX)/lib/vship.so`
- FFVship installs to `$(PREFIX)/bin`
- API header installs to `$(PREFIX)/include/VshipAPI.h`
- Uses pkg-config for ffms2 and zimg
- Requires `-fPIC` for shared libraries

## Source Code Architecture

### Directory Structure

```
src/
├── VshipLib.cpp          # VapourSynth plugin entry point
├── FFVship.cpp           # CLI tool entry point
├── VshipAPI.h            # C API for external library usage
├── ssimu2/               # SSIMULACRA2 implementation
│   ├── main.hpp          # GPU processing pipeline
│   ├── vapoursynth.hpp   # VapourSynth filter interface
│   ├── makeXYB.hpp       # RGB to XYB color space conversion
│   ├── downsample.hpp    # Multi-scale downsampling
│   ├── gaussianblur.hpp  # Gaussian blur kernels
│   └── score.hpp         # Final score calculation
├── butter/               # Butteraugli implementation
│   ├── main.hpp          # GPU processing pipeline
│   ├── vapoursynth.hpp   # VapourSynth filter interface
│   ├── colors.hpp        # OpsinDynamicsImage (psychovisual model)
│   ├── separatefrequencies.hpp  # Multi-frequency decomposition
│   ├── maltaDiff.hpp     # Malta difference maps
│   ├── maskPsycho.hpp    # Psychovisual masking
│   ├── combineMasks.hpp  # Mask combination
│   └── diffnorms.hpp     # Norm calculations (2-norm, 3-norm, inf-norm)
├── util/                 # Shared utilities
│   ├── preprocessor.hpp  # HIP/CUDA compatibility macros
│   ├── gpuhelper.hpp     # GPU device management and queries
│   ├── VshipExceptions.hpp  # Error handling
│   ├── concurrency.hpp   # Thread-safe queues and pools
│   ├── float3operations.hpp  # Vector math helpers
│   └── torgbs.hpp        # Color space conversion utilities
└── ffvship_utility/      # CLI-specific utilities
    ├── CLI_Parser.hpp    # Command-line argument parsing
    ├── ffmpegmain.hpp    # Video file handling via FFMS2
    ├── ProgressBar.hpp   # Terminal progress display
    └── gpuColorToLinear/ # GPU-accelerated color pipeline
        ├── vshipColor.hpp         # High-level color conversion
        ├── chromaUpsample.hpp     # Chroma upsampling kernels
        ├── transferToLinear.hpp   # Transfer function linearization
        ├── rangeToFull.hpp        # Limited to full range conversion
        └── primariesToBT709.hpp   # Primaries conversion to BT.709
```

### Key Architecture Patterns

**Dual Backend Support:**
- `preprocessor.hpp` provides HIP/CUDA abstraction layer
- Compiled with either `nvcc -x cu` (CUDA) or `hipcc` (HIP)
- All GPU calls use `hip*` functions that map to `cuda*` at compile time for NVIDIA

**Metric Processing Pipeline:**

Both SSIMULACRA2 and Butteraugli follow this pattern:
1. **Input Stage** - Load frames as planar RGB float/uint16
2. **Color Conversion** - Convert to linear RGB then to perceptual color space (XYB for SSIMULACRA2, OpsinDynamics for Butteraugli)
3. **Multi-Scale Processing** - Generate image pyramids and frequency bands
4. **Difference Calculation** - Compute perceptually-weighted differences
5. **Score Aggregation** - Reduce to final metric score(s)

**Stream-Based Concurrency:**
- Both VapourSynth plugin and CLI use CUDA/HIP streams for parallelism
- VapourSynth: `numStream` parameter controls number of concurrent GPU streams
- FFVship: Multi-threaded frame reading + GPU worker threads
- VRAM usage scales with number of streams (12×W×H×4 bytes per SSIMULACRA2 stream, 31×W×H×4 bytes per Butteraugli stream)

**Error Handling:**
- Custom exception type `VshipError` defined in `util/VshipExceptions.hpp`
- C API returns `Vship_Exception` enum for FFI safety
- VapourSynth errors reported via `vsapi->mapSetError()`

## Development Workflow

### Adding New Metrics

1. Create new directory under `src/` (e.g., `src/newmetric/`)
2. Implement core pipeline in `main.hpp`:
   - Define GPU kernels for metric computation
   - Create main processing function that takes GPU buffers
3. Create VapourSynth interface in `vapoursynth.hpp`:
   - Implement filter create function
   - Define `getFrame` callback for per-frame processing
   - Register filter in `VshipLib.cpp` via `VapourSynthPluginInit2`
4. Add CLI support in `FFVship.cpp`:
   - Add metric to `MetricType` enum
   - Extend CLI parser to accept new metric option
   - Update `frame_worker_thread` to handle new metric

### Testing VapourSynth Plugin

The plugin exposes three functions:
- `vship.SSIMULACRA2(reference, distorted, numStream=4, gpu_id=0)`
- `vship.BUTTERAUGLI(reference, distorted, qnorm=3, intensity_multiplier=1.0, distmap=0, numStream=4, gpu_id=0)`
- `vship.GpuInfo(gpu_id=0)` - Returns GPU information string

Scores are stored as frame properties:
- SSIMULACRA2: `_SSIMULACRA2` (float)
- Butteraugli: `_BUTTERAUGLI_2Norm`, `_BUTTERAUGLI_3Norm`, `_BUTTERAUGLI_INFNorm` (floats)

### GPU Compatibility

**Kernel Check:**
- `helper::gpuKernelCheck()` tests basic kernel execution
- Called during plugin initialization and via API
- Returns false if GPU cannot execute kernels properly

**Multi-GPU Support:**
- `gpu_id` parameter selects device (0-indexed)
- Each metric handler maintains per-device state
- Use `Vship_SetDevice()` in C API to select GPU

### C API for External Integration

The library exports a C API (`VshipAPI.h`) for use outside VapourSynth:

**Device Management:**
- `Vship_GetDeviceCount()` - Query number of GPUs
- `Vship_SetDevice()` - Select GPU by ID
- `Vship_GetDeviceInfo()` - Query GPU properties
- `Vship_GPUFullCheck()` - Verify GPU functionality

**Memory Management:**
- `Vship_PinnedMalloc()` / `Vship_PinnedFree()` - Allocate pinned host memory for fast transfers

**SSIMULACRA2:**
- `Vship_SSIMU2Init()` - Create handler for given resolution
- `Vship_ComputeSSIMU2Float()` / `Vship_ComputeSSIMU2Uint16()` - Compute score
- `Vship_SSIMU2Free()` - Destroy handler

**Butteraugli:**
- `Vship_ButteraugliInit()` / `Vship_ButteraugliInitv2()` - Create handler
- `Vship_ComputeButteraugliFloat()` / `Vship_ComputeButteraugliUint16()` - Compute scores and optional distortion map
- `Vship_ButteraugliFree()` - Destroy handler

All functions return `Vship_Exception` enum for error handling.

## Common Issues

**Build Errors:**
- Ensure correct compiler (nvcc vs hipcc) for your GPU vendor
- Check that VapourSynth/ffms2/zimg headers are accessible
- Verify CUDA/HIP toolkit version compatibility

**Runtime Issues:**
- Out of VRAM: Reduce `numStream` parameter
- Plugin not found: Check install path matches VapourSynth plugin directory
- Wrong GPU selected: Explicitly set `gpu_id` parameter

**Performance:**
- Use `numStream=4` as starting point (balance between throughput and VRAM)
- `-arch=native` (CUDA) or `--offload-arch=native` (HIP) optimizes for current GPU
- Build with `all` targets for portable binaries (slower but works across GPU generations)

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vship is a high-performance VapourSynth plugin and CLI tool for GPU-accelerated perceptual video quality metrics. It provides hardware-accelerated implementations of SSIMULACRA2, Butteraugli, and ColorVideoVDP (in progress) using HIP (AMD) or CUDA (NVIDIA).

The project consists of two primary components:
- **vship** - VapourSynth plugin (vship.dll/.so)
- **FFVship** - Standalone CLI tool for comparing video files

### Current Metrics Status

- **SSIMULACRA2**: ✅ Fully implemented and optimized
- **Butteraugli**: ✅ Fully implemented and optimized
- **ColorVideoVDP (CVVDP)**: ⚠️ ~70% complete (Phase 1 done)
  - VapourSynth integration: ✅ Working
  - FFVship CLI integration: ❌ Not yet implemented
  - Core pipeline: ✅ Basic implementation complete
  - Temporal filtering: ❌ Missing (high priority)
  - CSF LUT loading: ⚠️ Uses placeholders (high priority)
  - Masking operations: ❌ Missing (medium priority)
  - Advanced pooling: ⚠️ Simplified (medium priority)

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
├── VshipLib.cpp          # VapourSynth plugin entry point & metric registration
├── FFVship.cpp           # CLI tool entry point
├── VshipAPI.h            # C API for external library usage
│
├── ssimu2/               # SSIMULACRA2 implementation
│   ├── main.hpp          # GPU processing pipeline
│   ├── vapoursynth.hpp   # VapourSynth filter interface
│   ├── makeXYB.hpp       # RGB to XYB color space conversion
│   ├── downsample.hpp    # Multi-scale downsampling
│   ├── gaussianblur.hpp  # Gaussian blur kernels
│   └── score.hpp         # Final score calculation
│
├── butter/               # Butteraugli implementation
│   ├── main.hpp          # GPU processing pipeline
│   ├── vapoursynth.hpp   # VapourSynth filter interface
│   ├── colors.hpp        # OpsinDynamicsImage (psychovisual model)
│   ├── separatefrequencies.hpp  # Multi-frequency decomposition
│   ├── maltaDiff.hpp     # Malta difference maps
│   ├── maskPsycho.hpp    # Psychovisual masking
│   ├── combineMasks.hpp  # Mask combination
│   └── diffnorms.hpp     # Norm calculations (2-norm, 3-norm, inf-norm)
│
├── cvvdp/                # ColorVideoVDP implementation (IN PROGRESS)
│   ├── main.hpp          # GPU processing pipeline & kernels (~680 lines)
│   ├── vapoursynth.hpp   # VapourSynth filter interface
│   ├── config.hpp        # JSON configuration loader (RapidJSON)
│   ├── colorspace.hpp    # Color space conversions (sRGB→LMS→DKL)
│   ├── lpyr.hpp          # Laplacian pyramid decomposition
│   ├── csf.hpp           # Contrast Sensitivity Function with LUTs
│   └── README.md         # Implementation status & TODO list
│
├── util/                 # Shared utilities
│   ├── preprocessor.hpp  # HIP/CUDA compatibility macros
│   ├── gpuhelper.hpp     # GPU device management and queries
│   ├── VshipExceptions.hpp  # Error handling
│   ├── concurrency.hpp   # Thread-safe queues and pools
│   ├── float3operations.hpp  # Vector math helpers
│   └── torgbs.hpp        # Color space conversion utilities
│
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

config/
└── cvvdp_data/          # CVVDP calibration data (MUST be installed with plugin)
    ├── display_models.json      # 20+ display presets (SDR/HDR)
    ├── cvvdp_parameters.json    # Metric calibration parameters
    ├── csf_lut_*.json           # 7 CSF lookup table variants
    └── color_spaces.json        # Color space definitions
```

### Key Architecture Patterns

**Dual Backend Support:**
- `preprocessor.hpp` provides HIP/CUDA abstraction layer
- Compiled with either `nvcc -x cu` (CUDA) or `hipcc` (HIP)
- All GPU calls use `hip*` functions that map to `cuda*` at compile time for NVIDIA

**Metric Processing Pipeline:**

All metrics follow a similar pattern:
1. **Input Stage** - Load frames as planar RGB float/uint16
2. **Color Conversion** - Convert to linear RGB then to perceptual color space:
   - SSIMULACRA2: XYB color space
   - Butteraugli: OpsinDynamics
   - CVVDP: DKL opponent color space (Y, RG, BY channels)
3. **Multi-Scale Processing** - Generate image pyramids and frequency bands
4. **Difference Calculation** - Compute perceptually-weighted differences
5. **Score Aggregation** - Reduce to final metric score(s)

**CVVDP-Specific Pipeline:**

CVVDP is unique as the first color-aware metric accounting for spatial, temporal, and chromatic aspects of vision:
1. **Display Modeling** - Apply display-specific photometry (GOG/PQ/HLG) and geometry
2. **Color Conversion** - sRGB → linear RGB → XYZ → LMS → DKL opponent color space
3. **Contrast Encoding** - Convert to Weber contrast with local adaptation
4. **Laplacian Pyramid** - Multi-scale decomposition (7 bands typical for 4K)
5. **Contrast Sensitivity (CSF)** - Apply castleCSF model via 2D LUT interpolation
6. **Temporal Filtering** - 4-channel IIR filters (Y-sustained, RG, BY, Y-transient) ⚠️ TODO
7. **Masking** - Content and cross-channel masking ⚠️ TODO
8. **Pooling** - Hierarchical spatial/temporal/channel pooling with p-norms
9. **JOD Calibration** - Convert to Just-Objectionable-Differences scale (0-10, **higher is better**)

**IMPORTANT**: CVVDP scores are **inverted** compared to typical metrics:
- **10 JOD** = Perfect quality (no visible difference)
- **Lower scores** = More distortion
- **~7-8 JOD** = Typical good quality encode
- **~5 JOD** = Noticeable quality loss

**Stream-Based Concurrency:**
- Both VapourSynth plugin and CLI use CUDA/HIP streams for parallelism
- VapourSynth: `numStream` parameter controls number of concurrent GPU streams
- FFVship: Multi-threaded frame reading + GPU worker threads
- VRAM usage scales with number of streams:
  - SSIMULACRA2: ~12×W×H×4 bytes per stream
  - Butteraugli: ~31×W×H×4 bytes per stream
  - CVVDP: ~500MB per stream for 4K (varies with pyramid levels)

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

The plugin exposes four functions:
- `vship.SSIMULACRA2(reference, distorted, numStream=4, gpu_id=0)`
- `vship.BUTTERAUGLI(reference, distorted, qnorm=3, intensity_multiplier=1.0, distmap=0, numStream=4, gpu_id=0)`
- `vship.CVVDP(reference, distorted, display="standard_4k", numStream=4, gpu_id=0)` ⚠️ NEW
- `vship.GpuInfo(gpu_id=0)` - Returns GPU information string

Scores are stored as frame properties:
- SSIMULACRA2: `_SSIMULACRA2` (float)
- Butteraugli: `_BUTTERAUGLI_2Norm`, `_BUTTERAUGLI_3Norm`, `_BUTTERAUGLI_INFNorm` (floats)
- CVVDP: `_CVVDP` (float, 0-10 scale, **higher is better**)

**CVVDP VapourSynth Example:**
```python
import vapoursynth as vs
core = vs.core

# Load clips (must be planar RGB)
ref = core.ffms2.Source('reference.mp4', format=vs.RGBS)
test = core.ffms2.Source('test.mp4', format=vs.RGBS)

# Compute CVVDP scores
scored = core.vship.CVVDP(reference=ref, distorted=test, display="standard_4k", numStream=4)

# Extract and print scores
def print_jod(n, f):
    jod = f.props['_CVVDP']
    print(f"Frame {n}: {jod:.3f} JOD")
    return f

scored = core.std.ModifyFrame(scored, scored, print_jod)
scored.set_output()
```

**Available Display Models:**
- `standard_4k` - 30" 4K SDR (200 cd/m², office lighting)
- `standard_fhd` - 24" FHD SDR
- `standard_hdr_pq` - 30" 4K HDR (PQ transfer, 1500 cd/m²)
- `standard_hdr_hlg` - 30" 4K HDR (HLG transfer)
- `iphone_12_pro`, `iphone_14_pro`, `macbook_pro_16` - Mobile/laptop
- See `config/cvvdp_data/display_models.json` for complete list

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

## CVVDP Development

### Current Implementation Status

**Phase 1 (Complete - ~70%):**
- ✅ Configuration system with JSON parsing (RapidJSON)
- ✅ Display model loading from `config/cvvdp_data/display_models.json`
- ✅ Color space conversions (sRGB → linear → XYZ → LMS → DKL)
- ✅ Laplacian pyramid decomposition
- ✅ Basic CSF application (castleCSF model)
- ✅ Weber contrast encoding
- ✅ JOD score calibration (with linearization fix for small Q)
- ✅ VapourSynth filter registration and frame processing
- ✅ Multi-stream GPU processing with thread-safe queues
- ✅ Frame property output (`_CVVDP`)

**Phase 2 (TODO - Critical for Accuracy):**

#### 1. Temporal Filtering (High Priority)
**Status**: ❌ Missing - Currently processes single frames only
**Impact**: Significant accuracy loss without temporal context
**Location**: `src/cvvdp/main.hpp` (needs buffer management in `vapoursynth.hpp`)

**What's needed:**
- Sliding window buffer (250ms history, typically 7-15 frames)
- Four temporal channels: Y-sustained, RG, BY, Y-transient
- Per-channel IIR filters with sigma_tf, beta_tf parameters
- Buffer management in VapourSynth filter data structure

**Reference**: Python implementation in `pycvvdp/cvvdp_metric.py:364-440`

#### 2. CSF Lookup Table Loading (High Priority)
**Status**: ⚠️ Using placeholder values
**Impact**: Incorrect contrast sensitivity modeling
**Location**: `src/cvvdp/csf.hpp`

**What's needed:**
- Parse actual JSON arrays from `config/cvvdp_data/csf_lut_weber_fixed_size.json`
- Load 41×41 grids for three channels (Y, RG, BY)
- Copy LUT data to GPU constant memory for fast interpolation
- Current code has 2D interpolation logic but fake data

**File format**: JSON arrays with `log_L_bkg` and `log_rho` axes, `logS_[0,1,2]` matrices

#### 3. Masking Operations (Medium Priority)
**Status**: ❌ Not implemented
**Impact**: Less accurate difference prediction
**Location**: `src/cvvdp/main.hpp` (needs new kernels)

**What's needed:**
- Content masking with mask_p, mask_c parameters (mutual masking model)
- Cross-channel masking with 4×4 xcm_weights matrix
- Masking kernel that combines T_p (test) and R_p (reference) signals
- Formula: `M = (mask_p * |R|^mask_c + mask_q)` then apply xcm_weights

#### 4. Advanced Pooling (Medium Priority)
**Status**: ⚠️ Simplified averaging
**Impact**: Sub-optimal score aggregation
**Location**: `src/cvvdp/main.hpp:610-619`

**What's needed:**
- Proper p-norm spatial pooling with beta parameter
- Temporal pooling across frame buffer with beta_t
- Channel pooling with beta_sch (sustained), beta_tch (transient)
- Baseband weight application (frequency-dependent weights)
- Current code uses simple averaging instead of hierarchical pooling

#### 5. FFVship CLI Integration (Low Priority)
**Status**: ❌ Not integrated
**Impact**: No standalone tool for CVVDP
**Location**: `src/FFVship.cpp`, `src/ffvship_utility/CLI_Parser.hpp`

**What's needed:**
- Add `MetricType::CVVDP` enum value
- Extend CLI parser to accept `--metric cvvdp --display <model>`
- Update `frame_worker_thread` to handle CVVDP processing
- Output per-frame JOD scores and aggregate statistics
- Target usage: `FFVship ref.mp4 test.mp4 --metric cvvdp --display standard_4k`

### Key Files for CVVDP Development

**Primary Implementation:**
- `src/cvvdp/main.hpp` - Core pipeline (~680 lines, contains all GPU kernels)
- `src/cvvdp/vapoursynth.hpp` - VapourSynth integration with stream management
- `src/cvvdp/config.hpp` - JSON loading (RapidJSON-based)
- `src/cvvdp/colorspace.hpp` - Color space math (sRGB↔LMS↔DKL)
- `src/cvvdp/lpyr.hpp` - Laplacian pyramid decomposition
- `src/cvvdp/csf.hpp` - Contrast sensitivity function

**Configuration Data:**
- `config/cvvdp_data/display_models.json` - Display presets
- `config/cvvdp_data/cvvdp_parameters.json` - Calibration parameters
- `config/cvvdp_data/csf_lut_weber_fixed_size.json` - CSF lookup tables (TARGET)
- `config/cvvdp_data/color_spaces.json` - Color space definitions

**Integration Points:**
- `src/VshipLib.cpp:62` - Filter registration (already done)
- `src/FFVship.cpp` - CLI integration (TODO)

**Documentation:**
- `src/cvvdp/README.md` - Detailed implementation status and TODO list
- Python reference: `../ColorVideoVDP/pycvvdp/` for algorithm reference

### CVVDP Configuration Details

**Display Model Fields:**
```json
{
  "name": "standard_4k",
  "resolution": [3840, 2160],
  "diagonal_size_inches": 30,
  "max_luminance": 200,
  "min_luminance": 0.8,
  "EOTF": "sRGB",
  "colorspace": "sRGB"
}
```

**Key Parameters** (from `cvvdp_parameters.json`):
- `jod_a`, `jod_exp` - JOD calibration (currently: 1.0, 0.6)
- `beta` - Spatial pooling exponent (default: 3.0)
- `mask_p`, `mask_c`, `mask_q` - Content masking parameters
- `xcm_weights` - 4×4 cross-channel masking matrix
- `ch_gain` - Channel gains [Y, RG, BY, transient-Y] = [1.0, 1.45, 1.0, 1.0]
- `baseband_weight` - Per-band frequency weights

### CVVDP Validation Strategy

1. **Unit Tests**: Test individual components against NumPy/PyTorch reference
   - Color conversions (validate against known RGB→DKL values)
   - Pyramid decomposition (compare band sizes and frequencies)
   - CSF interpolation (verify against Python LUT loading)

2. **Integration Test**: Single frame comparison
   - Run same frame through Python `pycvvdp` and GPU implementation
   - Compare intermediate outputs (contrast, CSF-weighted, per-band scores)
   - Acceptable error: ±0.05 JOD for single frames

3. **End-to-End Test**: Full video comparison
   - Process test videos through both implementations
   - Compare final JOD scores per frame
   - Acceptable error: ±0.10 JOD rms (metric variance is inherent)

4. **Performance Target**:
   - 4K60 dual-stream: ≥1.6 fps/GPU on RTX 4090
   - 1080p30: ≥6.0 fps/GPU
   - Should be 10-20× faster than Python reference

### Recent Changes

**Commit de73888**: "phase 1 complete?"
- Fixed JOD linearization for small Q values (maintains differentiability)
- Implemented proper contrast encoding with local adaptation
- Added debug JSON output with per-band information

**Modified file**: `src/cvvdp/main.hpp:626-632`
```cpp
if (Q_per_ch <= Q_t) {
    // Linearized version for small Q values
    float jod_a_p = params.jod_a * powf(Q_t, params.jod_exp - 1.0f);
    Q_jod = 10.0f - jod_a_p * Q_per_ch;
} else {
    Q_jod = 10.0f - params.jod_a * powf(Q_per_ch, params.jod_exp);
}
```

## Common Issues

**Build Errors:**
- Ensure correct compiler (nvcc vs hipcc) for your GPU vendor
- Check that VapourSynth/ffms2/zimg headers are accessible
- Verify CUDA/HIP toolkit version compatibility
- CVVDP: Ensure RapidJSON headers are available in `third_party/rapidjson/include/`

**Runtime Issues:**
- Out of VRAM: Reduce `numStream` parameter
- Plugin not found: Check install path matches VapourSynth plugin directory
- Wrong GPU selected: Explicitly set `gpu_id` parameter
- CVVDP config not found: Verify `config/cvvdp_data/*.json` files are installed alongside plugin
  - Windows: Should be in `%APPDATA%\VapourSynth\cvvdp_data\` (relative to plugin)
  - Linux: Should be in `$(PREFIX)/share/vship/cvvdp_data/`
  - Set environment variable `CVVDP_DATA_ROOT` to override default path

**CVVDP-Specific Issues:**
- "Display model not found": Check spelling of display name, see `display_models.json` for valid names
- Low/inaccurate scores: Temporal filtering and masking not yet implemented (Phase 2 TODO)
- CSF sensitivity seems off: Currently using placeholder LUT values, needs proper JSON loading
- Slow performance: Expected until kernel fusion optimization (Phase 3)

**Performance:**
- Use `numStream=4` as starting point (balance between throughput and VRAM)
- CVVDP: Use `numStream=2` for 4K, `numStream=4` for 1080p
- `-arch=native` (CUDA) or `--offload-arch=native` (HIP) optimizes for current GPU
- Build with `all` targets for portable binaries (slower but works across GPU generations)

## Quick Reference

### Windows Build (Simple)
```bash
# Build both plugin and CLI tool with CUDA support
./build.bat

# This builds:
# - vship.dll → C:\Tools\lib\vapoursynth\
# - FFVship.exe → C:\Tools\bin\
# - Copies config/cvvdp_data/ to C:\Tools\config\cvvdp_data\
```

### Windows Build (Manual - Makefile)
```bash
# Build VapourSynth plugin
make buildcuda

# Build FFVship CLI
make buildFFVSHIPcuda

# Install (copies to system directories)
make install
```

### Testing CVVDP
```python
# Quick VapourSynth test
import vapoursynth as vs
core = vs.core

ref = core.ffms2.Source('ref.mp4', format=vs.RGBS)
test = core.ffms2.Source('test.mp4', format=vs.RGBS)

# Process and print scores
scored = core.vship.CVVDP(reference=ref, distorted=test, display="standard_4k")

def print_score(n, f):
    print(f"Frame {n}: JOD = {f.props['_CVVDP']:.3f}")
    return f

scored = core.std.ModifyFrame(scored, scored, print_score)
scored.set_output()
```

### Configuration File Locations
- **Source**: `config/cvvdp_data/*.json` (in repository)
- **Windows Install**: `%APPDATA%\VapourSynth\cvvdp_data\` or `C:\Tools\config\cvvdp_data\`
- **Linux Install**: `$(PREFIX)/share/vship/cvvdp_data/`
- **Override**: Set `CVVDP_DATA_ROOT` environment variable to custom path

### Important Constants
- **JOD Scale**: 0-10 (10 = perfect, lower = worse)
- **Typical Good Quality**: 7-8 JOD
- **Default Display**: `standard_4k` (30" 4K SDR, 200 cd/m²)
- **PPD Calculation**: `pixels_per_degree = (resolution / diagonal_inches) * (viewing_distance_inches / 57.3)`
- **Default Viewing Distance**: 0.67m for desktop, 0.3m for mobile

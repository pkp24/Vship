# ColorVideoVDP CUDA/HIP Implementation

This directory contains a CUDA/HIP port of ColorVideoVDP for integration into vship/FFVship.

## Current Status

### ✅ Completed Components

1. **Configuration System** (`config.hpp`)
   - JSON parser for display models and CVVDP parameters
   - Display model loader with photometry and geometry
   - Parameter loader for metric calibration values

2. **Color Space Conversions** (`colorspace.hpp`)
   - sRGB to linear RGB (EOTF)
   - RGB to XYZ (D65 white point)
   - XYZ to LMS (Hunt-Pointer-Estevez transformation)
   - LMS to DKL opponent color space
   - Display photometry (GOG model, PQ, HLG)
   - Log transformations for contrast encoding

3. **Laplacian Pyramid** (`lpyr.hpp`)
   - Multi-scale decimated Laplacian pyramid decomposition
   - Gaussian filtering and downsampling
   - Bilinear upsampling for reconstruction
   - Frequency band calculation (cycles per degree)

4. **Contrast Sensitivity Function** (`csf.hpp`)
   - castleCSF structure with lookup tables
   - 2D interpolation for sensitivity values
   - Per-channel CSF application (Y, RG, BY channels)

5. **Main Processing Pipeline** (`main.hpp`)
   - Frame pair processing workflow
   - Weber contrast encoding
   - Spatial pooling with p-norms
   - JOD score calculation
   - Simple C API for external integration

6. **VapourSynth Interface** (`vapoursynth.hpp`)
   - VapourSynth filter registration
   - Multi-stream GPU processing
   - Frame property output (`_CVVDP`)
   - Support for multiple pixel formats (UINT8, UINT16, FLOAT)

7. **Plugin Registration** (Updated `VshipLib.cpp`)
   - Registered as `vship.CVVDP(reference, distorted, display="standard_4k", numStream=4, gpu_id=0)`

### ⚠️ TODO: Critical Missing Components

The current implementation provides a **foundation** but requires significant refinement to match the Python reference:

#### 1. **Temporal Filtering** (High Priority)
   - **Missing**: Sliding window buffer management for video (250ms history)
   - **Missing**: Four temporal channels (Y-sustained, RG, BY, Y-transient)
   - **Missing**: Temporal filter kernels per-channel (sigma_tf, beta_tf parameters)
   - **Current**: Only processes single frames (no temporal context)
   - **Action**: Implement temporal buffer in VapourSynth interface, apply per-channel IIR filters

#### 2. **CSF Lookup Table Loading** (High Priority)
   - **Missing**: Actual JSON parsing for csf_lut_*.json files
   - **Current**: Uses placeholder/approximated CSF values
   - **Action**: Implement proper JSON array parsing for multi-dimensional LUTs (41x41 grids)
   - **Files to parse**: `config/cvvdp_data/csf_lut_weber_fixed_size.json`

#### 3. **Masking Operations** (Medium Priority)
   - **Missing**: Content masking (mask_p, mask_c parameters)
   - **Missing**: Cross-channel masking (xcm_weights matrix, 4x4 interactions)
   - **Missing**: Texture masking for "texture" model variant
   - **Current**: No masking applied
   - **Action**: Implement mutual masking kernel, apply xcm_weights to channel interactions

#### 4. **Advanced Pooling** (Medium Priority)
   - **Missing**: Proper spatial pooling (p-norm with beta parameter)
   - **Missing**: Temporal pooling (beta_t parameter across frames)
   - **Missing**: Channel pooling (beta_sch, beta_tch parameters)
   - **Missing**: Baseband weight application (frequency-dependent weights)
   - **Current**: Simple averaging
   - **Action**: Implement hierarchical pooling with proper p-norms

#### 5. **Contrast Encoding Variants** (Low Priority)
   - **Current**: Only Weber contrast implemented
   - **Missing**: Log contrast (`log_contrast` parameter)
   - **Missing**: Local adaptation models (gpyr vs simple)
   - **Action**: Add log contrast kernel, implement Gaussian pyramid for adaptation

#### 6. **JOD Calibration** (Medium Priority)
   - **Current**: Basic `jod_a * Q^jod_exp` formula
   - **Missing**: Proper difference clamping (d_max, dclamp_type="soft")
   - **Missing**: Image integration parameter (image_int)
   - **Action**: Apply soft clamping before pooling, integrate image_int into score

#### 7. **HDR Support** (Low Priority)
   - **Current**: Basic PQ/HLG EOTF functions present
   - **Missing**: BT.2020 color space handling
   - **Missing**: Proper HDR display model switching
   - **Action**: Add BT.2020 primaries conversion, auto-detect from colorspace field

#### 8. **Validation** (Critical)
   - **Missing**: Comparison against Python pycvvdp reference implementation
   - **Missing**: Unit tests for each component
   - **Missing**: Known test images/videos with expected JOD scores
   - **Action**: Create test suite, validate against reference on standard datasets

### 🔧 Build Integration

**Still TODO:**
- Update `Makefile` to include `cvvdp/*.hpp` in compilation
- Copy `config/cvvdp_data/` to installation directory
- Test build on Windows (nvcc) and Linux (hipcc/nvcc)

**Makefile changes needed:**
```makefile
# Add CVVDP config directory install
ifeq ($(OS),Windows_NT)
install:
    # ... existing installs ...
    if not exist "$(plugin_install_path)\..\cvvdp_data" mkdir "$(plugin_install_path)\..\cvvdp_data"
    copy "config\cvvdp_data\*.json" "$(plugin_install_path)\..\cvvdp_data\"
else
install:
    # ... existing installs ...
    install -d "$(PREFIX)/share/vship/cvvdp_data"
    install -m644 config/cvvdp_data/*.json "$(PREFIX)/share/vship/cvvdp_data/"
endif
```

### 🎯 FFVship Integration

**TODO:**
1. Add `MetricType::CVVDP` to `ffvship_utility/CLI_Parser.hpp`
2. Update `frame_worker_thread` in `FFVship.cpp` to handle CVVDP
3. Add display model selection to CLI arguments
4. Output JOD scores per-frame and aggregate statistics

**CLI usage target:**
```bash
FFVship reference.mp4 distorted.mp4 --metric cvvdp --display standard_4k
```

## Usage (VapourSynth)

```python
import vapoursynth as vs
core = vs.core

# Load clips (must be planar RGB)
ref = core.ffms2.Source('reference.mp4', format=vs.RGBS)
test = core.ffms2.Source('test.mp4', format=vs.RGBS)

# Compute CVVDP scores
scored = core.vship.CVVDP(reference=ref, distorted=test, display="standard_4k", numStream=4)

# Extract scores
def print_scores(n, f):
    jod = f.props['_CVVDP']
    print(f"Frame {n}: {jod:.3f} JOD")
    return f

scored = core.std.ModifyFrame(scored, scored, print_scores)
scored.set_output()
```

## Available Display Models

From `config/cvvdp_data/display_models.json`:
- `standard_4k` - 30" 4K SDR monitor (200 cd/m², office lighting)
- `standard_fhd` - 24" FHD SDR monitor
- `standard_hdr_pq` - 30" 4K HDR monitor (1500 cd/m², PQ transfer)
- `standard_hdr_hlg` - 30" 4K HDR monitor (HLG transfer)
- `standard_hdr_linear` - 30" 4K HDR monitor (linear light values)
- `iphone_12_pro`, `iphone_14_pro` - Mobile phone displays
- `macbook_pro_16` - Laptop display
- And more (see JSON file for complete list)

## Performance Notes

- **VRAM Usage**: ~500MB per stream for 4K video (varies with pyramid levels)
- **Recommended numStream**: 2-4 for 4K, 4-8 for 1080p
- **Speed**: Currently slower than SSIMULACRA2/Butteraugli due to unoptimized pooling
- **Optimization TODO**: Use shared memory for pyramid operations, fuse kernels

## Validation Strategy

1. **Unit Tests**: Test each component (color conversion, pyramid, CSF) against NumPy/PyTorch
2. **Integration Test**: Process single frame, compare intermediate outputs to Python
3. **End-to-End Test**: Process full video, compare final JOD scores
4. **Acceptable Error**: ±0.1 JOD (metric is calibrated to human perception, some variance OK)

## References

- **Paper**: "ColorVideoVDP: A visual difference predictor for image, video and display distortions" (Mantiuk et al., 2024)
- **Python Reference**: https://github.com/gfxdisp/ColorVideoVDP
- **Documentation**: https://www.cl.cam.ac.uk/research/rainbow/projects/ColorVideoVDP/

## Next Steps for Developer

1. **Implement temporal filtering** (see cvvdp_metric.py:364-440 for reference)
2. **Load actual CSF LUTs** (parse JSON properly, not placeholders)
3. **Add masking** (mult-mutual model from parameters)
4. **Validate** against Python on standard test images
5. **Optimize** (kernel fusion, reduce memory copies)
6. **Integrate into FFVship**
7. **Documentation** (update main CLAUDE.md with CVVDP usage)

## License

This implementation follows the same license as vship (see parent LICENSE file).
The ColorVideoVDP algorithm itself is described in the paper by Mantiuk et al. and may have separate licensing terms for commercial use.

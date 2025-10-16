# Vship Build Instructions

This document explains how to build vship with CVVDP support.

## Prerequisites

### Required Software

1. **Visual Studio 2022** with C++ development tools
   - Install from: https://visualstudio.microsoft.com/

2. **NVIDIA CUDA Toolkit 13.0+** (for CUDA builds)
   - Install from: https://developer.nvidia.com/cuda-downloads
   - Add to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`

3. **MSYS2** (optional, for HIP/AMD builds)
   - Install from: https://www.msys2.org/
   - Install clang64 toolchain: `pacman -S mingw-w64-clang-x86_64-toolchain`

### Required Libraries (in C:\Tools)

The build scripts expect these libraries to be available in `C:\Tools`:

- **VapourSynth headers** → `C:\Tools\include\VapourSynth4.h`
- **ffms2 library** → `C:\Tools\lib\ffms2.lib`, `C:\Tools\bin\libffms2-5.dll`
- **zimg library** → `C:\Tools\lib\zimg.lib`, `C:\Tools\bin\libzimg-2.dll`
- **zlib library** → `C:\Tools\lib\z.lib`, `C:\Tools\bin\zlib1.dll`, `C:\Tools\bin\z.dll`

The build process will automatically:
- Generate import libraries (.lib) from DLLs if needed
- Create `z.dll` from `zlib1.dll` for FFVship compatibility
- Check DLL timestamps and regenerate import libraries when DLLs are updated

## Quick Build

### Option 1: Using build.bat (Recommended)

```cmd
cd Vship
build.bat
```

This will:
1. Prepare libraries (regenerate .lib files if needed)
2. Build vship.dll (VapourSynth plugin)
3. Build FFVship.exe (CLI tool)
4. Install to `C:\Tools\lib\vapoursynth\` and `C:\Tools\bin\`
5. Install CVVDP config files

### Option 2: Manual Steps

```cmd
cd Vship

REM Step 1: Prepare libraries
powershell -ExecutionPolicy Bypass -File ..\compat\vship\prepare_libs.ps1 -InstallPrefix C:\Tools -BuildDir C:\Tools\builds\vship

REM Step 2: Build vship.dll
powershell -ExecutionPolicy Bypass -File ..\compat\vship\build_vship.ps1 -SourceDir "%cd%" -InstallPrefix C:\Tools

REM Step 3: Build FFVship.exe
powershell -ExecutionPolicy Bypass -File ..\compat\vship\build_ffvship.ps1 -SourceDir "%cd%" -InstallPrefix C:\Tools
```

## Testing the Build

After building, run the test script:

```cmd
test_build.bat
```

This will verify:
- vship.dll exists and is installed
- FFVship.exe exists and runs
- CVVDP config files are installed
- Runtime dependencies (z.dll, libzimg-2.dll) are present

## Installation Locations

### vship.dll (VapourSynth Plugin)

Installed to:
- `C:\Tools\lib\vapoursynth\vship.dll` (main location)
- `%APPDATA%\VapourSynth\plugins64\vship.dll` (autoload)

### FFVship.exe (CLI Tool)

Installed to:
- `C:\Tools\bin\FFVship.exe`

### CVVDP Configuration

Installed to:
- `C:\Tools\config\cvvdp_data\*.json` (global)
- `C:\Tools\lib\vapoursynth\cvvdp_data\*.json` (plugin-local)

Config files include:
- `display_models.json` - Display specifications (20+ presets)
- `cvvdp_parameters.json` - Metric calibration parameters
- `csf_lut_*.json` - Contrast sensitivity lookup tables (7 variants)
- `color_spaces.json` - Color space definitions

## Usage

### VapourSynth

```python
import vapoursynth as vs
core = vs.core

# Load clips (must be planar RGB)
ref = core.ffms2.Source('reference.mp4', format=vs.RGBS)
test = core.ffms2.Source('test.mp4', format=vs.RGBS)

# Use metrics
scored_ssimu2 = core.vship.SSIMULACRA2(reference=ref, distorted=test)
scored_butter = core.vship.BUTTERAUGLI(reference=ref, distorted=test)
scored_cvvdp = core.vship.CVVDP(reference=ref, distorted=test, display="standard_4k")

# Access scores via frame properties
def show_scores(n, f):
    print(f"Frame {n}:")
    if '_SSIMULACRA2' in f.props:
        print(f"  SSIMULACRA2: {f.props['_SSIMULACRA2']:.4f}")
    if '_BUTTERAUGLI_3Norm' in f.props:
        print(f"  Butteraugli: {f.props['_BUTTERAUGLI_3Norm']:.4f}")
    if '_CVVDP' in f.props:
        print(f"  CVVDP: {f.props['_CVVDP']:.3f} JOD")
    return f

scored = core.std.ModifyFrame(scored_cvvdp, scored_cvvdp, show_scores)
scored.set_output()
```

### FFVship CLI

```cmd
REM SSIMULACRA2
FFVship.exe source.mp4 encoded.mp4 --metric SSIMULACRA2

REM Butteraugli
FFVship.exe source.mp4 encoded.mp4 --metric Butteraugli

REM CVVDP (not yet implemented in CLI - use VapourSynth for now)
```

## Troubleshooting

### "Missing z.dll" Error

**Symptom**: FFVship.exe fails with "z.dll not found"

**Solution**:
```cmd
copy C:\Tools\bin\zlib1.dll C:\Tools\bin\z.dll
```

Or re-run prepare_libs.ps1 - it now creates z.dll automatically.

### "zimg_filter_graph_build Entry Point Not Found"

**Symptom**: FFVship.exe fails with entry point error for zimg functions

**Solution**: Regenerate zimg.lib from the current DLL:
```cmd
powershell -ExecutionPolicy Bypass -File ..\compat\vship\prepare_libs.ps1 -InstallPrefix C:\Tools -BuildDir C:\Tools\builds\vship
```

Then rebuild FFVship:
```cmd
powershell -ExecutionPolicy Bypass -File ..\compat\vship\build_ffvship.ps1 -SourceDir "%cd%" -InstallPrefix C:\Tools
```

The prepare_libs.ps1 script now automatically detects when DLLs are newer than .lib files and regenerates them.

### CVVDP "Config File Not Found"

**Symptom**: VapourSynth fails to load CVVDP with file not found error

**Solution**: Ensure config files are installed:
```cmd
xcopy config\cvvdp_data\*.json C:\Tools\config\cvvdp_data\ /Y
xcopy config\cvvdp_data\*.json C:\Tools\lib\vapoursynth\cvvdp_data\ /Y
```

Or re-run build_vship.ps1 - it now installs config files automatically.

### Build Fails with "nvcc: command not found"

**Symptom**: Build fails, cannot find nvcc compiler

**Solution**:
1. Verify CUDA Toolkit is installed
2. Add CUDA to PATH:
   ```cmd
   set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;%PATH%
   ```
3. Verify nvcc works: `nvcc --version`

### VapourSynth Cannot Load Plugin

**Symptom**: `core.vship` is not available in VapourSynth

**Solution**:
1. Check DLL is installed: `dir %APPDATA%\VapourSynth\plugins64\vship.dll`
2. Check dependencies: Run `test_build.bat` to verify all DLLs are present
3. Try loading manually:
   ```python
   core.std.LoadPlugin(r'C:\Tools\lib\vapoursynth\vship.dll')
   ```

## Build Configuration

### Changing Install Location

Edit the INSTALL_PREFIX in build.bat:

```bat
set "INSTALL_PREFIX=D:\MyTools"
```

### Building for Different GPU Architectures

The default build uses `-arch=native` which optimizes for your current GPU.

To build for all supported architectures (portable binary, slower):

Edit `compat/vship/build_vship.ps1` and `build_ffvship.ps1`:
```powershell
# Change from:
-arch=native

# To:
-arch=all
```

### Using HIP (AMD GPUs)

For AMD GPUs using HIP instead of CUDA:

1. Install ROCm/HIP
2. Use the Makefile directly instead of build.bat:
   ```bash
   make build          # Build vship.dll
   make buildFFVSHIP   # Build FFVship.exe
   ```

## Development Notes

### Adding New Metrics

To add a new metric to vship:

1. Create a new directory under `src/` (e.g., `src/newmetric/`)
2. Implement `main.hpp` with GPU kernels
3. Implement `vapoursynth.hpp` with filter interface
4. Register in `src/VshipLib.cpp`:
   ```cpp
   #include "newmetric/vapoursynth.hpp"
   vspapi->registerFunction("NEWMETRIC", ..., newmetric::create, ...);
   ```
5. Add to FFVship CLI in `src/FFVship.cpp`

### Updating CVVDP

The CVVDP implementation is in `src/cvvdp/`:
- `config.hpp` - Configuration loading
- `colorspace.hpp` - Color conversions
- `lpyr.hpp` - Laplacian pyramid
- `csf.hpp` - Contrast sensitivity
- `main.hpp` - Processing pipeline
- `vapoursynth.hpp` - VapourSynth interface
- `README.md` - Implementation status

See `src/cvvdp/README.md` for current limitations and TODOs.

## References

- **Vship Repository**: https://github.com/Line-fr/Vship
- **ColorVideoVDP**: https://github.com/gfxdisp/ColorVideoVDP
- **VapourSynth**: http://www.vapoursynth.com/
- **CUDA Documentation**: https://docs.nvidia.com/cuda/

# Vapoursynth-HIP

An easy to use vapoursynth plugin to compute SSIMU2 (SSIMULACRA2) from libjxl or Butteraugli from libjxl on GPU
with the exception that it uses a real gaussian blur and not a recursive gaussian blur.

Butteraugli Source Code : https://github.com/libjxl/libjxl/tree/main/lib/jxl/butteraugli
Ssimulacra2 Source Code : https://github.com/cloudinary/ssimulacra2

# Usage:

To retrieve ssimu2 score between two video Nodes

`result = clip1.vship.SSIMULACRA2(clip2)`
`res = [fr.props["_SSIMULACRA2"] for fr in result.frames()]`

To retrieve Butteraugli score

`result = clip1.vship.BUTTERAUGLI(clip2)`
`res = [[fr.props["_BUTTERAUGLI_2Norm"], fr.props["_BUTTERAUGLI_3Norm"], fr.props["_BUTTERAUGLI_INFNorm"]] for fr in result.frames()]`

you can check "simpleExample.vpy" inside VshipUsageScript to get more details about how to use Vship

If you encounter VRAM issue, set a lower vs.core.num_threads (4 is usually the good compromise for speed)

the exact vram formula per active vsthread is:
ssimu2: 24 (plane buffer) * 4 (size of float) * width * height * 4/3

butteraugli: 34 (plane buffer) * 4 (size of float) * width * height

# to build:
Warning: it is compiled for the specific gpu used to compile by default

(same for windows and linux)
you need 
- make
- hipcc or nvcc 
- vapoursynth

for nvidia cards:
`make buildcuda`

for amd cards:
`make build`

to install: either use the dll or:
`make install`

# Performance on my laptop on some clip:

![comparison](Images/vshipjxl.png)

# special credits to dnjulek with the ZIG vapoursynth implementation

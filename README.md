# Vapoursynth-HIP

An easy to use vapoursynth plugin to compute SSIMU2 (SSIMULACRA2) or Butteraugli on GPU

# Usage:

you can check "simpleExample.vpy" inside VshipUsageScript to learn how to use Vship

you can try to tune the number of vapoursynth threads
-> more threads can sometimes be faster but with more vram usage (for 1080p it can go up to 300 VRAM MB per threads for ssimu2 and butteraugli)
`vs.core.num_threads = ?`

the exact vram formula is:
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

# Vapoursynth-HIP-SSIMU2
An easy to use plugin for vapoursynth performing SSIMU2 measurments using the GPU with HIP

# Usage:

`vs.core.vship.SSIMULACRA2(original, distorded)`

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
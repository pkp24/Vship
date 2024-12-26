# Vapoursynth-HIP-SSIMU2
An easy to use plugin for vapoursynth performing SSIMU2 measurments using the GPU with HIP

# Usage:

`vs.core.vship.SSIMULACRA2(original, distorded)`

to convert to RGBS:

`vclip = vclip.resize.Bicubic(height=vclip.height, width=vclip.width, format=vs.RGBS, matrix_in_s="709", transfer_in_s="srgb", transfer_s="linear")`

to get the resulting values (can be modified but this is an example):

`res = [[ind, fr.props["_SSIMULACRA2"]] for (ind, fr) in enumerate(vclip.frames())]`

you can try to tune the number of vapoursynth threads
-> more threads is faster but with more vram
`vs.core.threads = ?`

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

# special credits to dnjulek with the ZIG vapoursynth implementation
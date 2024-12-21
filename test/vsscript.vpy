import vapoursynth as vs
import os

vs.core.num_threads = 12
vclip = vs.core.lsmas.LWLibavSource(r"../../videoexample.mkv")

vclip = vclip.resize.Bicubic(height=vclip.height, width=vclip.width, format=vs.RGBS, matrix_in_s="709", transfer_in_s="srgb", transfer_s="linear")

vs.core.std.LoadPlugin("../hipssimu2.dll")

vclip = vs.core.vship.SSIMULACRA2(vclip, vclip)
vclip.set_output()
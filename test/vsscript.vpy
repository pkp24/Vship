import vapoursynth as vs
import os

vclip = vs.core.lsmas.LWLibavSource(r"../../videoexample.mkv")

vclip = vclip.resize.Bicubic(height=vclip.height, width=vclip.width, format=vs.RGBS, matrix_in_s="709", transfer_in_s="srgb", transfer_s="linear")

vs.core.std.LoadPlugin("../hipssimu2.dll")

out = vs.core.vship.SSIMULACRA2(vclip, vclip)
out.set_output()
import vapoursynth as vs
import os

vs.core.std.LoadPlugin("../hipssimu2.dll")

vs.core.num_threads = 30
vclip = vs.core.lsmas.LWLibavSource(r"../../videoexample.mkv")[1000:2000]

vclip = vclip.resize.Bicubic(format=vs.RGBS, matrix_in_s='709').fmtc.transfer(transs="srgb", transd="linear", bits=32)
vclip = vs.core.vship.SSIMULACRA2(vclip, vclip)
vclip.set_output()
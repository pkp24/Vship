import vapoursynth as vs
import os

vclip = vs.core.bs.VideoSource(r"../../videoexample.mp4")

vs.core.std.LoadPlugin("../hipssimu2.dll")

out = vs.core.vship.SSIMULACRA2(vclip, vclip)
out.set_output()
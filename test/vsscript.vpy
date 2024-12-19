import vapoursynth as vs
import os

vclip = vs.core.bs.VideoSource(r"C:\Users\leofr\Desktop\Passable Female Voice - Comprehensive Lesson.mp4")

vs.core.std.LoadPlugin("../build/hipssimu2.dll")

out = vs.core.vshipssimu2.SSIMULACRA2(vclip, vclip)
out.set_output()
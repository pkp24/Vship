import vapoursynth as vs
import os

vclip = vs.core.bs.VideoSource(r"C:\Users\leofr\Desktop\Passable Female Voice - Comprehensive Lesson.mp4")

vs.core.std.LoadPlugin("../build/hipssimu2.dll")

out = vs.core.invert.Filter(vclip, 50)
out.set_output()
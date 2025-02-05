import vapoursynth as vs

#using bestsource to open video
clip1 = vs.core.bs.VideoSource("MyVideoPath")
clip2 = vs.core.bs.VideoSource("MyVideoPath2")

clip1 = clip1.resize.Bicubic(height=clip1.height, width=clip1.width, format=vs.RGBS)
clip2 = clip2.resize.Bicubic(height=clip1.height, width=clip1.width, format=vs.RGBS) #resize to size of clip1 and goes to RGBS

#use one or the other
result = clip1.vship.BUTTERAUGLI(clip1, clip2)
#result = clip1.vship.SSIMULACRA2(clip1, clip2)

#it will contain all the ssimulacra2 values computed
res = [[ind, fr.props["_SSIMULACRA2"]] for (ind, fr) in enumerate(vclip.frames())]

#for butteraugli you can use one of these if you wish (it is also possible to get all 3 in one pass if you do it another way)
#res = [[ind, fr.props["_BUTTERAUGLI_2Norm"]] for (ind, fr) in enumerate(vclip.frames())]
#res = [[ind, fr.props["_BUTTERAUGLI_3Norm"]] for (ind, fr) in enumerate(vclip.frames())]
#res = [[ind, fr.props["_BUTTERAUGLI_INFNorm"]] for (ind, fr) in enumerate(vclip.frames())]
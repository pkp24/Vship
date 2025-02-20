import vapoursynth as vs

vs.core.num_threads = 4 #can be increased/tuned for better performance and lowered to reduce VRAM

#using bestsource to open video
clip1 = vs.core.bs.VideoSource("MyVideoPath")
clip2 = vs.core.bs.VideoSource("MyVideoPath2")

#make sure that the 2 have the same height and width or resize them

#use one or the other
#result = clip1.vship.BUTTERAUGLI(clip2, intensity_multiplier = 80, distmap = 0)
#intensity_multiplier is an optional argument
#distmap = 1 allows retrieving a grayscale image representing the distorsion map from butter

result = clip1.vship.SSIMULACRA2(clip2)

#it will contain all the ssimulacra2 values computed
res = [[ind, fr.props["_SSIMULACRA2"]] for (ind, fr) in enumerate(result.frames())]

#for butteraugli you can use one of these if you wish (it is also possible to get all 3 in one pass if you do it another way)
#res = [[ind, fr.props["_BUTTERAUGLI_2Norm"]] for (ind, fr) in enumerate(result.frames())]
#res = [[ind, fr.props["_BUTTERAUGLI_3Norm"]] for (ind, fr) in enumerate(result.frames())]
#res = [[ind, fr.props["_BUTTERAUGLI_INFNorm"]] for (ind, fr) in enumerate(result.frames())]

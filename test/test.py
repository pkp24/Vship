#dependency check
try:
    import vapoursynth as vs
except:
    print("Failed to import vapoursynth")

try:
    core = vs.core
except:
    print("Failed to get core from vapoursynth. You may not have installed vapoursynth python lib")

if hasattr(core, "bs"):
    importer = core.bs.VideoSource
elif hasattr(core, "ffms2"):
    importer = core.ffms2.Source
else:
    print("No supported importer found")
    exit()

if (not hasattr(core, "vship")):
    print("vship was not found so we cannot test it")
    exit()

from subprocess import call
import os
import time

#video link list
videos = [
    'https://videos.pexels.com/video-files/3446616/3446616-uhd_2560_1440_25fps.mp4',
    'https://videos.pexels.com/video-files/8348731/8348731-uhd_2560_1440_25fps.mp4',
    'https://videos.pexels.com/video-files/4153409/4153409-uhd_2560_1440_25fps.mp4',
    'https://videos.pexels.com/video-files/18947687/18947687-sd_640_360_30fps.mp4',
]

#import videos
if (not os.path.isdir("videos")):
    os.mkdir("videos")
testnum = 0
for el in videos:
    out = os.path.join("videos", f"sample{testnum}.{el[-3:]}")
    if (not os.path.isfile(out)):
        try:
            call(f"wget {el} -O {out}", shell=True)
        except:
            print("Failed to download ", el, "with commandline : ", f"wget {el} -O {out}")
            exit()
    testnum += 1

#create distorded
def distorded_filtering1(source):
    return core.std.BoxBlur(source)

disfilters = [distorded_filtering1]

#compute scores
vshipScores = []
for (i, el) in enumerate(os.listdir("videos")):
    ref = importer(os.path.join("videos", el))
    vshipScores.append([])
    for disf in disfilters:
        dis = disf(ref)

        core.num_threads = 4

        temp = core.vship.SSIMULACRA2(ref, dis)
        init = time.time()
        SSIMU2score = [fr.props["_SSIMULACRA2"] for fr in temp.frames()]
        SSIMU2time = time.time()-init

        temp = core.vship.BUTTERAUGLI(ref, dis)
        
        init = time.time()
        BUTTERscore = [[fr.props["_BUTTERAUGLI_2Norm"], fr.props["_BUTTERAUGLI_3Norm"], fr.props["_BUTTERAUGLI_INFNorm"]] for fr in temp.frames()]
        BUTTERtime = time.time()-init

        vshipScores[-1] = [SSIMU2time, SSIMU2score, BUTTERtime, BUTTERscore]

core.num_threads = 24

vszipScores = []
if hasattr(core, "vszip"):
    for (i, el) in enumerate(os.listdir("videos")):
        ref = importer(os.path.join("videos", el))
        vszipScores.append([])
        for disf in disfilters:
            dis = disf(ref)

            temp = core.vszip.Metrics(ref, dis)
            init = time.time()
            SSIMU2score = [fr.props["_SSIMULACRA2"] for fr in temp.frames()]
            SSIMU2time = time.time()-init
            vszipScores[-1] = [SSIMU2time, SSIMU2score]

jxlScores = []
if hasattr(core, "julek"):
    for (i, el) in enumerate(os.listdir("videos")):
        ref = importer(os.path.join("videos", el))
        jxlScores.append([])
        for disf in disfilters:
            dis = disf(ref)

            src = vs.core.resize.Bicubic(ref, format=vs.RGBS, matrix_in=1)
            dst = vs.core.resize.Bicubic(dis, format=vs.RGBS, matrix_in=1)

            temp = core.julek.SSIMULACRA(src, dst, feature=0)
            init = time.time()
            SSIMU2score = [fr.props["_SSIMULACRA2"] for fr in temp.frames()]
            SSIMU2time = time.time()-init

            temp = core.julek.Butteraugli(src, dst)
            
            init = time.time()
            BUTTERscore = [fr.props["_FrameButteraugli"] for fr in temp.frames()]
            BUTTERtime = time.time()-init

            jxlScores[-1] = [SSIMU2time, SSIMU2score, BUTTERtime, BUTTERscore]
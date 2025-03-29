from Lav1e import SSIMU2Score, vs
import time
from matplotlib import pyplot
import os
from subprocess import call

if (not os.path.isfile("sample.y4m")): 
    try:
        import wget
        wget.download("https://media.xiph.org/video/derf/y4m/factory_1080p30.y4m", out="sample.y4m")
    except:
        print("wget api not found, let s try your shell")
        call("wget https://media.xiph.org/video/derf/y4m/factory_1080p30.y4m -O sample.y4m", shell=True)
if (not os.path.isfile("testscores.mkv")): call("ffmpeg -i sample.y4m -crf 25 -an -sn testscores.mkv", shell=True)

score = SSIMU2Score()
source = r"sample.y4m" #for input file test
dis = r"testscores.mkv" #for distorded file

begin = 100
end = 1100

MainVSThreads = 24
vshipnumStream = 8

gpu_id = 0

vs.core.num_threads = MainVSThreads
print("starting compute")
init = time.time()
score.compute_butter(source, dis, 1, begin, end, "jxl")
print(score)
print("jxl butter had ", (end-begin)/(time.time()-init), "fps")
jxlbutterfps = (end-begin)/(time.time()-init)//0.1 * 0.1
yjxl = [el[1] for el in score.scores]
print("---------------------------------------")
init = time.time()
score.compute_butter(source, dis, 1, begin, end, "vship", numStream=vshipnumStream, gpu_id=gpu_id)
print(score)
print("vship butter had ", (end-begin)/(time.time()-init), "fps")
vshipbutterfps = (end-begin)/(time.time()-init)//0.1 * 0.1
yvship = [el[1] for el in score.scores]

print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")

init = time.time()
score.compute(source, dis, 1, begin, end, "jxl")
print(score)
print("jxl ssimu2 had ", (end-begin)/(time.time()-init), "fps")
jxlssimu2fps = (end-begin)/(time.time()-init)//0.1 * 0.1
yjxls = [el[1] for el in score.scores]
print("---------------------------------------")
init = time.time()
score.compute(source, dis, 1, begin, end, "vship", numStream=vshipnumStream, gpu_id=gpu_id)
print(score)
print("vship ssimu2 had ", (end-begin)/(time.time()-init), "fps")
vshipssimu2fps = (end-begin)/(time.time()-init)//0.1 * 0.1
yvships = [el[1] for el in score.scores]
print("---------------------------------------")
init = time.time()
score.compute(source, dis, 1, begin, end, "vszip")
print(score)
print("vszip ssimu2 had ", (end-begin)/(time.time()-init), "fps")
vszipssimu2fps = (end-begin)/(time.time()-init)//0.1 * 0.1
yvszips = [el[1] for el in score.scores]

x = list(range((end-begin)))

#for i in x:
#	if abs(yjxl[i]-yvship[i]) > 0.8:
#		print("big issue at ", i)

pyplot.style.use('dark_background')
fig, axs = pyplot.subplots(2, 1)
axs[0].plot(x, yjxl, label=f"jxl butter {jxlbutterfps} fps", color="white")
axs[0].plot(x, yvship, label=f"vship butter {vshipbutterfps} fps", color="yellow")
axs[0].set_ylabel("butter")
axs[0].set_ylim(bottom = 0)
axs[0].legend()

axs[1].plot(x, yjxls, label=f"jxl ssimu2 {jxlssimu2fps} fps", color="white")
axs[1].plot(x, yvszips, label=f"vszip ssimu2 {vszipssimu2fps} fps", color="red")
axs[1].plot(x, yvships, label=f"vship ssimu2 {vshipssimu2fps} fps", color="yellow")
axs[1].set_xlabel("frame")
axs[1].set_ylabel("ssim2")
axs[1].set_ylim(top = 100)
axs[1].legend()

fig.suptitle("Vship vs jxl on ryzen 7900x + RX 7900XTX")
pyplot.legend()
pyplot.show()
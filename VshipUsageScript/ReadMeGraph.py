from Lav1e import SSIMU2Score
import time
from matplotlib import pyplot

score = SSIMU2Score()
source = r"test.mkv" #for input file test
dis = r"testscores.mkv" #for distorded file

init = time.time()
score.compute_butter(source, dis, 1, 1000, 1200, "jxl")
print(score)
print("jxl butter had ", 200/(time.time()-init), "fps")
jxlbutterfps = 200/(time.time()-init)//0.1 * 0.1
yjxl = [el[1] for el in score.scores]
print("---------------------------------------")
init = time.time()
score.compute_butter(source, dis, 1, 1000, 1200, "vship")
print(score)
print("vship butter had ", 200/(time.time()-init), "fps")
vshipbutterfps = 200/(time.time()-init)//0.1 * 0.1
yvship = [el[1] for el in score.scores]

print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")

init = time.time()
score.compute(source, dis, 1, 1000, 1200, "jxl")
print(score)
print("jxl ssimu2 had ", 200/(time.time()-init), "fps")
jxlssimu2fps = 200/(time.time()-init)//0.1 * 0.1
yjxls = [el[1] for el in score.scores]
print("---------------------------------------")
init = time.time()
score.compute(source, dis, 1, 1000, 1200, "vship")
print(score)
print("vship ssimu2 had ", 200/(time.time()-init), "fps")
vshipssimu2fps = 200/(time.time()-init)//0.1 * 0.1
yvships = [el[1] for el in score.scores]

x = list(range(200))

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
axs[1].plot(x, yvships, label=f"vship ssimu2 {vshipssimu2fps} fps", color="yellow")
axs[1].set_xlabel("frame")
axs[1].set_ylabel("ssim2")
axs[1].set_ylim(top = 100)
axs[1].legend()

fig.suptitle("Vship vs jxl on ryzen 7940HS AVX512 + RTX 4050 mobile")
pyplot.legend()
pyplot.show()
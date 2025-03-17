#dependency analyzer
from sys import exit
import os
import json
import time
try:
	from matplotlib import pyplot
except:
	print("warning: matplotlib is not installed, running pip install matplotlib")
	check_output("pip install matplotlib", stderr = DEVNULL)
	try:
		from matplotlib import pyplot
	except:
		print("fails to install matplotlib")
		exit(1)

#vapoursynth dependency checks will not be able to solve anything but only signal problems for now
try:
	import vapoursynth as vs
except:
	print("You need to install vapoursynth...")
	exit(1)

if not "bs" in dir(vs.core):
	print("You must install BestSource to use the script")
	exit(1)
if not "julek" in dir(vs.core):
	print("you must install julek plugin to use the script")
	exit(1)
if not "vship" in dir(vs.core):
	print("you must install vship plugin to use the script")
	exit(1)

print("dependency check passed")

class SSIMU2Score:
	def __init__(self):
		self.scores = []
		self.source = "None"
		self.distorded = "None"
		self.type = "SSIMU2"

	def compute(self, originalFile : str, distordedFile : str, skip : int = 5, begin : int = 0, end : int = None, method:str="vship") -> None:
		src =  (vs.core.bs.VideoSource(originalFile) if (type(originalFile) == str) else originalFile)[begin:end:skip]
		dis = (vs.core.bs.VideoSource(distordedFile) if (type(distordedFile) == str) else distordedFile)[begin:end:skip]

		dis = vs.core.resize.Bicubic(dis, format=vs.RGBS, matrix_in=1)
		if (src.width == dis.width and src.height == dis.height):
			src = vs.core.resize.Bicubic(src, format=vs.RGBS, matrix_in=1)
		else:
			src = vs.core.resize.Bicubic(src, format=vs.RGBS, matrix_in=1, width=dis.width, height=dis.height)

		if method == "vship":
			result = src.vship.SSIMULACRA2(dis)
		elif method == "vszip":
			result = src.vszip.Metrics(dis)
		elif method == "jxl":
			result = src.julek.SSIMULACRA(dis, feature=0)
		else:
			print("invalid method")

		res = [[begin + ind*skip, fr.props["_SSIMULACRA2"]] for (ind, fr) in enumerate(result.frames())]
		#res = [k for k in res if k[1] > 0]

		self.scores = res
		self.source = originalFile
		self.distorded = distordedFile
		self.type = "SSIMU2"

	def compute_butter(self, originalFile, distordedFile, skip : int = 5, begin : int = 0, end : int = None, method:str="vship") -> None:
		src =  (vs.core.bs.VideoSource(originalFile) if (type(originalFile) == str) else originalFile)[begin:end:skip]
		dis = (vs.core.bs.VideoSource(distordedFile) if (type(distordedFile) == str) else distordedFile)[begin:end:skip]

		dis = vs.core.resize.Bicubic(dis, format=vs.RGBS, matrix_in=1)
		if (src.width == dis.width and src.height == dis.height):
			src = vs.core.resize.Bicubic(src, format=vs.RGBS, matrix_in=1)
		else:
			src = vs.core.resize.Bicubic(src, format=vs.RGBS, matrix_in=1, width=dis.width, height=dis.height)

		if method == "jxl":
			result = src.julek.Butteraugli(dis)
			res = [[begin + ind*skip, fr.props["_FrameButteraugli"]] for (ind, fr) in enumerate(result.frames())]
		elif method == "vship":
			result = src.vship.BUTTERAUGLI(dis)
			res = [[begin + ind*skip, fr.props["_BUTTERAUGLI_INFNorm"]] for (ind, fr) in enumerate(result.frames())]

		#res = [k for k in res if k[1] > 0]

		self.scores = res
		self.source = originalFile
		self.distorded = distordedFile
		self.type = "Butter"

	def statistics(self) -> (int, int, int, int, int):
		#returns (avg, deviation, median, 5th percentile, 95th percentile)
		intlist = [k[1] for k in self.scores]
		avg = sum(intlist)/len(intlist)
		deviation = ((sum([k*k for k in intlist])/len(intlist)) - avg*avg)**0.5
		sortedlist = sorted(intlist)
		return (avg, deviation, sortedlist[len(intlist)//2], sortedlist[len(intlist)//20], sortedlist[19*len(intlist)//20])

	def __repr__(self):
		stats = self.statistics()
		res = f"{self.type} for files {self.source} and {self.distorded}\n"
		res += f"Mean : {stats[0]}\n"
		res += f"Standard deviation : {stats[1]}\n"
		res += f"Median : {stats[2]}\n"
		res += f"5th percentile : {stats[3]}\n"
		res += f"95th percentile : {stats[4]}\n"
		return res

	def save(self, savefile : str) -> None:
		with open(savefile, "w") as file:
			json.dump([self.scores, self.source, self.distorded, self.type], file)

	def load(self, savefile : str) -> None:
		with open(savefile, "r") as file:
			res = json.load(file)
		self.scores = res[0]
		self.source = res[1]
		self.distorded = res[2]
		self.type = res[3]

	def histogram(self):
		x = [k*0.5 for k in range(201)]
		y = [0 for k in x]
		for score in scores:
			if score >= 0 and score <= 100:
				y[int(score*2)] += 1
		pyplot.plot(x, y)
		pyplot.show()

if __name__ == "__main__":
	pass
	#input("done")
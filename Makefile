
current_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

ifeq ($(OS),Windows_NT)
	dllend := .dll
	fpiccuda := 
	fpicamd := 
	mvcommand := copy $(subst /,\,"$(current_dir)vship$(dllend)") "%APPDATA%\VapourSynth\plugins64"
else
	dllend := .so
	fpiccuda := -Xcompiler -fPIC
	fpicamd := -fPIC
	mvcommand := cp ./vship$(dllend) /usr/lib/vapoursynth
endif

.FORCE:

buildFFVSHIP: src/ffmpegmain.cpp .FORCE
	hipcc src/ffmpegmain.cpp --offload-arch=native -Wno-unused-result -Wno-ignored-attributes $(shell pkg-config --libs ffms2 zimg) -o FFVship

buildFFVSHIPcuda: src/ffmpegmain.cpp .FORCE
	nvcc -x cu src/ffmpegmain.cpp -arch=native $(subst -pthread,-Xcompiler="-pthread",$(shell pkg-config --libs ffms2 zimg)) -o FFVship

build: src/vapoursynthPlugin.cpp .FORCE
	hipcc src/vapoursynthPlugin.cpp --offload-arch=native -I "$(current_dir)include" -Wno-unused-result -Wno-ignored-attributes -shared $(fpicamd) -o "$(current_dir)vship$(dllend)"

buildcuda: src/vapoursynthPlugin.cpp .FORCE
	nvcc -x cu src/vapoursynthPlugin.cpp -arch=native -I "$(current_dir)include"  -shared $(fpiccuda) -o "$(current_dir)vship$(dllend)"

buildcudaall: src/vapoursynthPlugin.cpp .FORCE
	nvcc -x cu src/vapoursynthPlugin.cpp -arch=all -I "$(current_dir)include" -shared $(fpiccuda) -o "$(current_dir)vship$(dllend)"

buildall: src/vapoursynthPlugin.cpp .FORCE
	hipcc src/vapoursynthPlugin.cpp --offload-arch=gfx1100,gfx1101,gfx1102,gfx1103,gfx1030,gfx1031,gfx1032,gfx906,gfx801,gfx802,gfx803 -I "$(current_dir)include" -Wno-unused-result -Wno-ignored-attributes -shared $(fpicamd) -o "$(current_dir)vship$(dllend)"


install:
	$(mvcommand)

test: .FORCE build
	vspipe .\test\vsscript.vpy .

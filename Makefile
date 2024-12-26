
current_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

ifeq ($(OS),Windows_NT)
	dllend := .dll
	mvcommand := copy $(subst /,\,"$(current_dir)hipssimu2$(dllend)") "%APPDATA%\VapourSynth\plugins64"
else
	dllend := .so
	mvcommand := cp "$(current_dir)hipssimu2$(dllend)" /usr/lib/vapoursynth
endif

.FORCE:

build: src/main.cpp .FORCE
	hipcc src/main.cpp --offload-arch=native -I "$(current_dir)include" -Wno-ignored-attributes -shared -static -o "$(current_dir)hipssimu2$(dllend)"

buildcuda: src/main.cpp .FORCE
	nvcc -x cu src/main.cpp -arch=native -I "$(current_dir)include" -shared -o "$(current_dir)hipssimu2$(dllend)"

buildcudaall: src/main.cpp .FORCE
	nvcc -x cu src/main.cpp -arch=all -I "$(current_dir)include" -shared -o "$(current_dir)hipssimu2$(dllend)"

buildall: src/main.cpp .FORCE
	hipcc src/main.cpp --offload-arch=gfx1100,gfx1101,gfx1102,gfx1030,gfx1031,gfx1032,gfx906,gfx801,gfx802,gfx803 -I "$(current_dir)include" -Wno-ignored-attributes -shared -static -o "$(current_dir)hipssimu2$(dllend)"


install:
	$(mvcommand)

test: .FORCE build
	vspipe .\test\vsscript.vpy .
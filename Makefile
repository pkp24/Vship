current_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

PREFIX ?= /usr/local
DESTDIR ?=

ifeq ($(OS),Windows_NT)
    dllend := .dll
    fpiccuda :=
    fpicamd :=
    plugin_install_path := $(APPDATA)\VapourSynth\plugins64
    exe_install_path := $(ProgramFiles)\FFVship
else
    dllend := .so
    fpiccuda := -Xcompiler -fPIC
    fpicamd := -fPIC
    plugin_install_path := $(DESTDIR)$(PREFIX)/lib/vapoursynth
    exe_install_path := $(DESTDIR)$(PREFIX)/bin
endif

.FORCE:

buildFFVSHIP: src/ffmpegmain.cpp .FORCE
	hipcc src/ffmpegmain.cpp --offload-arch=native -Wno-unused-result -Wno-ignored-attributes $(shell pkg-config --libs ffms2 zimg) -o FFVship

buildFFVSHIPcuda: src/ffmpegmain.cpp .FORCE
	nvcc -x cu src/ffmpegmain.cpp -arch=native $(subst -pthread,-Xcompiler="-pthread",$(shell pkg-config --libs ffms2 zimg)) -o FFVship

buildFFVSHIPall: src/ffmpegmain.cpp .FORCE
	hipcc src/ffmpegmain.cpp --offload-arch=gfx1100,gfx1101,gfx1102,gfx1103,gfx1030,gfx1031,gfx1032,gfx906,gfx801,gfx802,gfx803 -Wno-unused-result -Wno-ignored-attributes $(shell pkg-config --libs ffms2 zimg) -o FFVship

buildFFVSHIPcudaall: src/ffmpegmain.cpp .FORCE
	nvcc -x cu src/ffmpegmain.cpp -arch=all $(subst -pthread,-Xcompiler="-pthread",$(shell pkg-config --libs ffms2 zimg)) -o FFVship

build: src/vapoursynthPlugin.cpp .FORCE
	hipcc src/vapoursynthPlugin.cpp --offload-arch=native -I "$(current_dir)include" -Wno-unused-result -Wno-ignored-attributes -shared $(fpicamd) -o "$(current_dir)vship$(dllend)"

buildcuda: src/vapoursynthPlugin.cpp .FORCE
	nvcc -x cu src/vapoursynthPlugin.cpp -arch=native -I "$(current_dir)include" -shared $(fpiccuda) -o "$(current_dir)vship$(dllend)"

buildcudaall: src/vapoursynthPlugin.cpp .FORCE
	nvcc -x cu src/vapoursynthPlugin.cpp -arch=all -I "$(current_dir)include" -shared $(fpiccuda) -o "$(current_dir)vship$(dllend)"

buildall: src/vapoursynthPlugin.cpp .FORCE
	hipcc src/vapoursynthPlugin.cpp --offload-arch=gfx1100,gfx1101,gfx1102,gfx1103,gfx1030,gfx1031,gfx1032,gfx906,gfx801,gfx802,gfx803 -I "$(current_dir)include" -Wno-unused-result -Wno-ignored-attributes -shared $(fpicamd) -o "$(current_dir)vship$(dllend)"

ifeq ($(OS),Windows_NT)
install:
	if exist "$(current_dir)vship$(dllend)" copy "$(current_dir)vship$(dllend)" "$(plugin_install_path)"
else
install:
	@if [ -f "$(current_dir)vship$(dllend)" ]; then \
		install -d "$(plugin_install_path)"; \
		install -m755 "$(current_dir)vship$(dllend)" "$(plugin_install_path)/vship$(dllend)"; \
	fi
	@if [ -f "FFVship" ]; then \
		install -d "$(exe_install_path)"; \
		install -m755 FFVship "$(exe_install_path)/FFVship"; \
	fi
endif

test: .FORCE build
	vspipe ./test/vsscript.vpy .

@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d "D:\Encoding_Research\Convert_CVVDP\Vship"
make buildcuda
make buildFFVSHIPcuda

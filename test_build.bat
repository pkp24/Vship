@echo off
REM Quick test to verify vship build is working

echo ========================================
echo Testing vship build
echo ========================================
echo.

REM Check if DLLs exist
set "INSTALL_PREFIX=C:\Tools"
set "VS_DLL=%INSTALL_PREFIX%\lib\vapoursynth\vship.dll"
set "FFVSHIP_EXE=%INSTALL_PREFIX%\bin\FFVship.exe"

echo Checking for built files...
if exist "%VS_DLL%" (
    echo [OK] vship.dll found: %VS_DLL%
) else (
    echo [FAIL] vship.dll NOT found: %VS_DLL%
    goto :error
)

if exist "%FFVSHIP_EXE%" (
    echo [OK] FFVship.exe found: %FFVSHIP_EXE%
) else (
    echo [FAIL] FFVship.exe NOT found: %FFVSHIP_EXE%
    goto :error
)
echo.

REM Check CVVDP config files
echo Checking CVVDP configuration...
set "CVVDP_CONFIG=%INSTALL_PREFIX%\config\cvvdp_data\display_models.json"
if exist "%CVVDP_CONFIG%" (
    echo [OK] CVVDP config found: %CVVDP_CONFIG%
) else (
    echo [WARN] CVVDP config not found at: %CVVDP_CONFIG%
    echo        (CVVDP metric will not work without config files)
)
echo.

REM Test FFVship help
echo Testing FFVship.exe...
"%FFVSHIP_EXE%" --help >nul 2>&1
if errorlevel 1 (
    echo [FAIL] FFVship.exe failed to run
    echo.
    echo Attempting to run with error output:
    "%FFVSHIP_EXE%" --help
    goto :error
) else (
    echo [OK] FFVship.exe runs successfully
)
echo.

REM Check dependencies
echo Checking runtime dependencies...
set "ZLIB=%INSTALL_PREFIX%\bin\zlib1.dll"
set "ZDLL=%INSTALL_PREFIX%\bin\z.dll"
set "ZIMG=%INSTALL_PREFIX%\bin\libzimg-2.dll"

if exist "%ZLIB%" (
    echo [OK] zlib1.dll found
) else (
    echo [WARN] zlib1.dll not found
)

if exist "%ZDLL%" (
    echo [OK] z.dll found
) else (
    echo [WARN] z.dll not found (needed for FFVship)
)

if exist "%ZIMG%" (
    echo [OK] libzimg-2.dll found
) else (
    echo [WARN] libzimg-2.dll not found (needed for FFVship)
)
echo.

echo ========================================
echo All tests passed!
echo ========================================
echo.
echo To use vship in VapourSynth:
echo   import vapoursynth as vs
echo   core = vs.core
echo   core.vship.SSIMULACRA2(ref, dist)
echo   core.vship.BUTTERAUGLI(ref, dist)
echo   core.vship.CVVDP(ref, dist, display="standard_4k")
echo.
echo To use FFVship:
echo   FFVship.exe source.mp4 encoded.mp4 --metric SSIMULACRA2
echo.
pause
exit /b 0

:error
echo.
echo ========================================
echo BUILD TEST FAILED
echo ========================================
echo.
echo Please run build.bat to build vship
pause
exit /b 1

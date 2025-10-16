@echo off
setlocal enabledelayedexpansion

REM Build script for vship with CVVDP integration
REM Uses the compat build scripts for proper dependency management

set "SOURCE_DIR=%~dp0"
set "INSTALL_PREFIX=C:\Tools"
set "BUILD_DIR=%INSTALL_PREFIX%\builds\vship"

REM Remove trailing backslash
if "%SOURCE_DIR:~-1%"=="\" set "SOURCE_DIR=%SOURCE_DIR:~0,-1%"

echo ========================================
echo Building vship with CVVDP
echo ========================================
echo Source: %SOURCE_DIR%
echo Install: %INSTALL_PREFIX%
echo Build: %BUILD_DIR%
echo.

REM Step 1: Prepare libraries (regenerate import libs if needed)
echo [1/3] Preparing libraries...
powershell.exe -ExecutionPolicy Bypass -File "..\compat\vship\prepare_libs.ps1" -InstallPrefix "%INSTALL_PREFIX%" -BuildDir "%BUILD_DIR%"
if errorlevel 1 (
    echo ERROR: Library preparation failed
    exit /b 1
)
echo.

REM Step 2: Build vship.dll (VapourSynth plugin)
echo [2/3] Skipped Building vship.dll with CVVDP support...
REM powershell.exe -ExecutionPolicy Bypass -File "..\compat\vship\build_vship.ps1" -SourceDir "%SOURCE_DIR%" -InstallPrefix "%INSTALL_PREFIX%"
REM if errorlevel 1 (
REM     echo ERROR: vship.dll build failed
REM     exit /b 1
REM )
echo.

REM Step 3: Build FFVship.exe (CLI tool)
echo [3/3] Building FFVship.exe...
powershell.exe -ExecutionPolicy Bypass -File "..\compat\vship\build_ffvship.ps1" -SourceDir "%SOURCE_DIR%" -InstallPrefix "%INSTALL_PREFIX%"
if errorlevel 1 (
    echo ERROR: FFVship.exe build failed
    exit /b 1
)
echo.

echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Outputs:
echo   - vship.dll     : %INSTALL_PREFIX%\lib\vapoursynth\vship.dll
echo   - FFVship.exe   : %INSTALL_PREFIX%\bin\FFVship.exe
echo.
echo CVVDP config    : %SOURCE_DIR%\config\cvvdp_data\*.json
echo.
echo VapourSynth usage:
echo   core.vship.SSIMULACRA2(ref, dist)
echo   core.vship.BUTTERAUGLI(ref, dist)
echo   core.vship.CVVDP(ref, dist, display="standard_4k")
echo.
echo FFVship usage:
echo   FFVship.exe source.mp4 encoded.mp4 --metric SSIMULACRA2
echo   FFVship.exe source.mp4 encoded.mp4 --metric Butteraugli
echo   (Note: CVVDP CLI support not yet implemented)
echo.

pause

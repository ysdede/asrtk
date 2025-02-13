@echo off
setlocal enabledelayedexpansion

set "INPUT_DIR=%~1"
set "OUTPUT_DIR=%~2"

if "%INPUT_DIR%"=="" (
    echo Input directory not specified
    exit /b 1
)

if "%OUTPUT_DIR%"=="" (
    echo Output directory not specified
    exit /b 1
)

echo Processing directories in: %INPUT_DIR%
echo Output directory: %OUTPUT_DIR%
echo.

for /d %%d in ("%INPUT_DIR%*") do (
    echo.
    echo Processing directory: %%~nxd
    set "CMD=asrtk split "%%d" "%OUTPUT_DIR%%%~nxd" --tolerance 500 -fm"
    echo Command: !CMD!
    !CMD!
)

echo Done.

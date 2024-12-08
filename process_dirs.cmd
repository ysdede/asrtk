@echo off
setlocal enabledelayedexpansion

set "INPUT_DIR=%~1"
set "OUTPUT_DIR=%~2"
set "KEEP_EFFECTS=%~3"
set "FORCE_MERGE=%~4"

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
echo Keep effects: %KEEP_EFFECTS%
echo Force merge: %FORCE_MERGE%
echo.

for /d %%d in ("%INPUT_DIR%*") do (
    echo.
    echo Processing directory: %%~nxd
    if "%KEEP_EFFECTS%"=="true" (
        if "%FORCE_MERGE%"=="true" (
            echo Command: asrtk split "%%d" "%OUTPUT_DIR%%%~nxd" --tolerance 500 --keep-effects -fm
            asrtk split "%%d" "%OUTPUT_DIR%%%~nxd" --tolerance 500 --keep-effects -fm
        ) else (
            echo Command: asrtk split "%%d" "%OUTPUT_DIR%%%~nxd" --tolerance 500 --keep-effects
            asrtk split "%%d" "%OUTPUT_DIR%%%~nxd" --tolerance 500 --keep-effects
        )
    ) else (
        if "%FORCE_MERGE%"=="true" (
            echo Command: asrtk split "%%d" "%OUTPUT_DIR%%%~nxd" --tolerance 500 -fm
            asrtk split "%%d" "%OUTPUT_DIR%%%~nxd" --tolerance 500 -fm
        ) else (
            echo Command: asrtk split "%%d" "%OUTPUT_DIR%%%~nxd" --tolerance 500
            asrtk split "%%d" "%OUTPUT_DIR%%%~nxd" --tolerance 500
        )
    )
)

echo Done.

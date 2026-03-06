@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ==========================================
echo   Voice Persona Starting...
echo ==========================================
echo.

:: Find conda activate script
set CONDA_BAT=
if exist "D:\miniconda3\Scripts\activate.bat" (
    set "CONDA_BAT=D:\miniconda3\Scripts\activate.bat"
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    set "CONDA_BAT=%USERPROFILE%\miniconda3\Scripts\activate.bat"
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    set "CONDA_BAT=%USERPROFILE%\anaconda3\Scripts\activate.bat"
) else if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
    set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
) else if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
    set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
)

if "%CONDA_BAT%"=="" (
    echo [WARNING] conda activate script not found. Trying PATH...
    call conda activate main
) else (
    call "%CONDA_BAT%" main
)

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to activate conda env "main"
    echo Please check: conda env list
    echo.
    pause
    exit /b 1
)

echo [OK] conda env "main" activated
echo.

:: Install electron dependencies if needed
if not exist "%~dp0electron\node_modules" (
    echo [INFO] Installing Electron dependencies...
    cd /d "%~dp0electron"
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed
        pause
        exit /b 1
    )
    cd /d "%~dp0"
    echo [OK] Electron dependencies installed
    echo.
)

:: Launch Electron (it will start the Python server internally)
cd /d "%~dp0electron"
echo [OK] Launching Electron app...
npx electron .

pause

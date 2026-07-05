@echo off
setlocal
rem One-command start on Windows (Docker Desktop):
rem   - double-click this file, or run: start.bat (cmd) / .\start.bat (PowerShell)
cd /d "%~dp0"

echo [1/3] Checking Docker (can take a minute if Docker Desktop just started)...
docker info >nul 2>&1
if not %errorlevel%==0 (
    echo.
    echo [!] Docker is not responding. Make sure Docker Desktop is installed and
    echo     running - wait until the whale icon stops animating - then run this
    echo     script again.
    pause
    exit /b 1
)

echo [2/3] Detecting NVIDIA GPU...
set GPU=0
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    docker image inspect alpine >nul 2>&1 || (
        echo     Downloading a small test image - one time only...
        docker pull alpine
    )
    docker run --rm --gpus all alpine true >nul 2>&1
    if not errorlevel 1 set GPU=1
)

echo [3/3] Building and starting WatchDog Cam...
echo     The FIRST build downloads a few GB - progress is shown below.
echo.
if "%GPU%"=="1" (
    echo     NVIDIA GPU detected - GPU acceleration ON
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
) else (
    echo     No NVIDIA GPU available - running in CPU mode
    docker compose up -d --build
)

echo.
echo Done. Open http://localhost:8000 in your browser.
pause

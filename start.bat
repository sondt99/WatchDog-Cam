@echo off
rem One-command start on Windows (Docker Desktop):
rem   - cmd:        start.bat
rem   - PowerShell: .\start.bat
rem Docker Desktop has NVIDIA GPU support built in (just needs the normal
rem NVIDIA driver) - this probes for it and picks the right mode.
cd /d "%~dp0"

docker info >nul 2>&1
if not %errorlevel%==0 (
    echo [!] Docker is not running. Start Docker Desktop first, wait for the
    echo     whale icon to turn steady, then run this script again.
    pause
    exit /b 1
)

docker run --rm --gpus all alpine true >nul 2>&1
if %errorlevel%==0 (
    echo NVIDIA GPU detected - starting with GPU acceleration
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
) else (
    echo No GPU available to Docker - starting in CPU mode
    docker compose up -d --build
)

echo.
echo Open http://localhost:8000
pause

@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"
echo ========================================
echo TTS-Story Setup
echo ========================================
echo.

REM Check Python installation
echo [1/10] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%

REM Create virtual environment
echo.
echo [2/10] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo.
echo [3/10] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo [4/10] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install PyTorch (let pip/PyTorch auto-detect CUDA)
echo.
echo [5/10] Installing PyTorch...
echo This may take several minutes...
echo.

REM Check if CUDA is available using Python
python -c "import sys; sys.exit(0)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not working in virtual environment
    pause
    exit /b 1
)

REM Install PyTorch with CUDA 12.1 (most compatible)
echo Installing PyTorch with CUDA 12.1 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if errorlevel 1 (
    echo.
    echo PyTorch installation failed, trying CPU version...
    pip install torch torchvision torchaudio
)

REM Install other dependencies (excluding torch + chatterbox runtime handled separately)
echo.
echo [6/10] Installing other Python dependencies...
findstr /v /i "torch" requirements.txt > temp_requirements.txt
pip install -r temp_requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
del temp_requirements.txt

REM Install local Chatterbox runtime
echo.
echo [7/10] Installing Chatterbox Turbo runtime...
pip install chatterbox-tts --no-deps
if errorlevel 1 (
    echo ERROR: Failed to install chatterbox-tts
    pause
    exit /b 1
)

REM Install VoxCPM runtime
echo.
echo [8/10] Installing VoxCPM 1.5 runtime...
pip install voxcpm --no-deps
if errorlevel 1 (
    echo WARNING: Failed to install voxcpm - VoxCPM engine will not be available
)

REM Optional performance extras (best-effort)
echo.
echo [9/10] Installing optional performance extras...
echo - flash-attn (Qwen3 speedup)
echo - hf_xet (faster Hugging Face downloads)
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo WARNING: torch not available in this environment. Skipping flash-attn install.
) else (
    if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        echo Using MSVC 2019 toolchain for CUDA compatibility...
        call "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    )
    where cl >nul 2>&1
    if errorlevel 1 (
        set "VS_INSTALL_DIR="
        set "VSWHERE_EXE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
        if exist "%VSWHERE_EXE%" (
            for /f "usebackq delims=" %%I in (`"%VSWHERE_EXE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -version "[16.0,17.0)" -property installationPath`) do (
                set "VS_INSTALL_DIR=%%I"
            )
            if "!VS_INSTALL_DIR!"=="" (
                for /f "usebackq delims=" %%I in (`"%VSWHERE_EXE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
                    set "VS_INSTALL_DIR=%%I"
                )
            )
        )
        if "!VS_INSTALL_DIR!"=="" (
            for %%D in (
                "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools"
                "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools"
                "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Community"
                "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Professional"
                "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Enterprise"
                "%ProgramFiles(x86)%\Microsoft Visual Studio\2026\BuildTools"
                "%ProgramFiles(x86)%\Microsoft Visual Studio\2026\Community"
                "%ProgramFiles(x86)%\Microsoft Visual Studio\2026\Professional"
                "%ProgramFiles(x86)%\Microsoft Visual Studio\2026\Enterprise"
                "%ProgramFiles%\Microsoft Visual Studio\18\Community"
                "%ProgramFiles%\Microsoft Visual Studio\18\BuildTools"
            ) do (
                if exist "%%~D\VC\Auxiliary\Build\vcvars64.bat" (
                    set "VS_INSTALL_DIR=%%~D"
                )
            )
        )
        if not "!VS_INSTALL_DIR!"=="" (
            if exist "!VS_INSTALL_DIR!\VC\Auxiliary\Build\vcvars64.bat" (
                echo Found MSVC Build Tools. Initializing build environment...
                call "!VS_INSTALL_DIR!\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
            )
        )
    )
    where cl >nul 2>&1
    if errorlevel 1 (
        echo WARNING: MSVC build tools not found. Skipping flash-attn install.
        echo To install MSVC Build Tools manually, download:
        echo https://aka.ms/vs/17/release/vs_buildtools.exe
        echo Then select "Desktop development with C++" and install.
    ) else (
        pip install wheel
        if errorlevel 1 (
            echo WARNING: Failed to install wheel. Skipping flash-attn install.
        ) else (
        set "DISTUTILS_USE_SDK=1"
        set "MSSdk=1"
        REM flash-attn needs torch available; disable build isolation to avoid missing torch in build env
        pip install flash-attn --no-build-isolation
        if errorlevel 1 (
            echo WARNING: flash-attn install failed. Qwen3 will use eager attention ^(slower^).
        )
        )
    )
)
pip install hf_xet
if errorlevel 1 (
    echo WARNING: hf_xet install failed. Hugging Face downloads may be slower.
)

call :EnsureVoicePromptFolder
call :InstallSox
call :InstallRubberBand
call :InstallFFmpeg

REM Check for espeak-ng
echo.
echo ========================================
echo Checking espeak-ng...
echo ========================================
where espeak-ng >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: espeak-ng not found!
    echo.
    echo Please install espeak-ng manually:
    echo 1. Download from: https://github.com/espeak-ng/espeak-ng/releases
    echo 2. Get the file: espeak-ng-X64.msi
    echo 3. Run the installer
    echo 4. Restart your terminal
    echo.
    echo The application will NOT work without espeak-ng!
    echo.
) else (
    echo espeak-ng is installed!
)

REM Verify installation
echo.
echo ========================================
echo Verifying Installation
echo ========================================
echo.
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('CUDA Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-only')"

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. If espeak-ng is not installed, install it now
echo 2. Run: run.bat
echo 3. Open browser to: http://localhost:5000
echo.
pause
goto :EOF

:EnsureVoicePromptFolder
set "VOICE_PROMPTS_DIR=%~dp0data\voice_prompts"
if not exist "%VOICE_PROMPTS_DIR%" (
    mkdir "%VOICE_PROMPTS_DIR%" >nul 2>&1
)
goto :EOF

:InstallSox
echo.
echo [10/10] Installing SoX...
set "SOX_DIR=%~dp0tools\sox"
set "SOX_URL_1=https://gigenet.dl.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2-win64.zip"
set "SOX_URL_2=https://gigenet.dl.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2-win32.zip"
set "SOX_EXE_URL=https://gigenet.dl.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2-win64.exe"
set "SOX_ZIP=%TEMP%\sox_download.zip"
set "SOX_EXTRACT=%TEMP%\sox_extract"

if exist "%SOX_DIR%\sox.exe" (
    echo SoX already present.
    goto :EOF
)

if not exist "%~dp0tools" (
    mkdir "%~dp0tools"
) >nul 2>&1

set "SOX_INSTALLED="
call :TrySoxDownload "%SOX_URL_1%"
if not defined SOX_INSTALLED call :TrySoxDownload "%SOX_URL_2%"

:AfterSoxDownload
if not defined SOX_INSTALLED (
    echo WARNING: SoX install failed. Attempting to download the official installer...
    if not exist "%SOX_DIR%" mkdir "%SOX_DIR%" >nul 2>&1
    set "SOX_SETUP_EXE=%SOX_DIR%\sox-14.4.2-win64.exe"
    powershell -NoLogo -NoProfile -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13; Invoke-WebRequest -Uri '%SOX_EXE_URL%' -OutFile '%SOX_SETUP_EXE%' -UseBasicParsing -ErrorAction Stop; } catch { Write-Error $_.Exception.Message; exit 1 }" >nul 2>&1
    if errorlevel 1 (
        echo WARNING: Failed to download SoX installer. Install manually from https://sourceforge.net/projects/sox/files/sox/
    ) else (
        echo SoX installer downloaded: %SOX_SETUP_EXE%
        echo Run it manually, then re-run setup.bat or run.bat.
    )
)

if exist "%SOX_ZIP%" del "%SOX_ZIP%" >nul 2>&1
if exist "%SOX_EXTRACT%" rmdir /s /q "%SOX_EXTRACT%" >nul 2>&1
goto :EOF

:TrySoxDownload
set "SOX_URL=%~1"
if exist "%SOX_ZIP%" del "%SOX_ZIP%" >nul 2>&1
if exist "%SOX_EXTRACT%" rmdir /s /q "%SOX_EXTRACT%" >nul 2>&1

echo Downloading SoX from %SOX_URL% ...
powershell -NoLogo -NoProfile -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13; Invoke-WebRequest -Uri '%SOX_URL%' -OutFile '%SOX_ZIP%' -UseBasicParsing -ErrorAction Stop; } catch { Write-Error $_.Exception.Message; exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Failed to download SoX from %SOX_URL%.
    goto :EOF
)

powershell -NoLogo -NoProfile -Command "try { if ((Get-Item '%SOX_ZIP%').Length -lt 102400) { exit 2 } } catch { exit 2 }" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Downloaded SoX archive looks invalid.
    goto :EOF
)

powershell -Command "Expand-Archive -Path '%SOX_ZIP%' -DestinationPath '%SOX_EXTRACT%' -Force" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Failed to extract SoX archive.
    goto :EOF
)

set "SOX_SOURCE="
for /f "delims=" %%F in ('dir /b /s "%SOX_EXTRACT%\sox.exe" 2^>nul') do (
    set "SOX_SOURCE=%%F"
    goto :FoundSox
)
:FoundSox
if not defined SOX_SOURCE (
    echo WARNING: sox.exe not found in downloaded archive.
    goto :EOF
)

for %%F in ("%SOX_SOURCE%") do set "SOX_SOURCE_DIR=%%~dpF"
if not defined SOX_SOURCE_DIR (
    echo WARNING: Unable to determine SoX source directory.
    goto :EOF
)

if exist "%SOX_DIR%" rmdir /s /q "%SOX_DIR%" >nul 2>&1
mkdir "%SOX_DIR%" >nul 2>&1
xcopy /E /I /Y "%SOX_SOURCE_DIR%*.*" "%SOX_DIR%\" >nul
if errorlevel 1 (
    echo WARNING: Failed to copy SoX files to tools directory.
    goto :EOF
)

echo SoX installed to %SOX_DIR%.
set "SOX_INSTALLED=1"
goto :EOF

:InstallRubberBand
echo.
echo [11/10] Installing Rubber Band CLI...
set "RB_DIR=%~dp0tools\rubberband"
set "RB_URL=https://breakfastquay.com/files/releases/rubberband-4.0.0-gpl-executable-windows.zip"
set "RB_ZIP=%TEMP%\rubberband_cli.zip"
set "RB_EXTRACT=%TEMP%\rubberband_cli_extract"

if exist "%RB_DIR%\rubberband.exe" (
    echo Rubber Band CLI already present.
    goto :EOF
)

if not exist "%~dp0tools" (
    mkdir "%~dp0tools"
) >nul 2>&1

if exist "%RB_ZIP%" del "%RB_ZIP%" >nul 2>&1
if exist "%RB_EXTRACT%" rmdir /s /q "%RB_EXTRACT%" >nul 2>&1

echo Downloading Rubber Band CLI from: %RB_URL%
powershell -NoLogo -NoProfile -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13; Invoke-WebRequest -Uri '%RB_URL%' -OutFile '%RB_ZIP%' -UseBasicParsing -ErrorAction Stop; } catch { Write-Error $_.Exception.Message; exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Failed to download Rubber Band CLI. FX will fall back to basic processing.
    goto :EOF
)

powershell -Command "Expand-Archive -Path '%RB_ZIP%' -DestinationPath '%RB_EXTRACT%' -Force" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Failed to extract Rubber Band CLI archive.
    goto :EOF
)

set "RB_SOURCE="
for /f "delims=" %%F in ('dir /b /s "%RB_EXTRACT%\rubberband.exe" 2^>nul') do (
    set "RB_SOURCE=%%F"
    goto :FoundRB
)
:FoundRB
if not defined RB_SOURCE (
    echo WARNING: rubberband.exe not found in downloaded archive.
    goto :EOF
)

for %%F in ("%RB_SOURCE%") do set "RB_SOURCE_DIR=%%~dpF"
if not defined RB_SOURCE_DIR (
    echo WARNING: Unable to determine Rubber Band source directory.
    goto :EOF
)

if exist "%RB_DIR%" rmdir /s /q "%RB_DIR%" >nul 2>&1
mkdir "%RB_DIR%" >nul 2>&1
xcopy /E /I /Y "%RB_SOURCE_DIR%*.*" "%RB_DIR%\" >nul
if errorlevel 1 (
    echo WARNING: Failed to copy Rubber Band CLI files to tools directory.
    goto :EOF
)

echo Rubber Band CLI installed to %RB_DIR%.

if exist "%RB_ZIP%" del "%RB_ZIP%" >nul 2>&1
if exist "%RB_EXTRACT%" rmdir /s /q "%RB_EXTRACT%" >nul 2>&1
goto :EOF

:InstallFFmpeg
echo.
echo [12/10] Installing FFmpeg...
set "FF_DIR=%~dp0tools\ffmpeg"
set "FF_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
set "FF_ZIP=%TEMP%\ffmpeg_download.zip"
set "FF_EXTRACT=%TEMP%\ffmpeg_extract"

if exist "%FF_DIR%\ffmpeg.exe" (
    echo FFmpeg already present.
    goto :EOF
)

if not exist "%~dp0tools" (
    mkdir "%~dp0tools"
) >nul 2>&1

if exist "%FF_ZIP%" del "%FF_ZIP%" >nul 2>&1
if exist "%FF_EXTRACT%" rmdir /s /q "%FF_EXTRACT%" >nul 2>&1

echo Downloading FFmpeg from GitHub...
powershell -NoLogo -NoProfile -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13; Invoke-WebRequest -Uri '%FF_URL%' -OutFile '%FF_ZIP%' -UseBasicParsing -ErrorAction Stop; } catch { Write-Error $_.Exception.Message; exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Failed to download FFmpeg. Audio merging may use system FFmpeg.
    goto :EOF
)

echo Extracting FFmpeg...
powershell -Command "Expand-Archive -Path '%FF_ZIP%' -DestinationPath '%FF_EXTRACT%' -Force" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Failed to extract FFmpeg archive.
    goto :EOF
)

set "FF_SOURCE="
for /f "delims=" %%F in ('dir /b /s "%FF_EXTRACT%\ffmpeg.exe" 2^>nul') do (
    set "FF_SOURCE=%%F"
    goto :FoundFF
)
:FoundFF
if not defined FF_SOURCE (
    echo WARNING: ffmpeg.exe not found in downloaded archive.
    goto :EOF
)

for %%F in ("%FF_SOURCE%") do set "FF_SOURCE_DIR=%%~dpF"
if not defined FF_SOURCE_DIR (
    echo WARNING: Unable to determine FFmpeg source directory.
    goto :EOF
)

if exist "%FF_DIR%" rmdir /s /q "%FF_DIR%" >nul 2>&1
mkdir "%FF_DIR%" >nul 2>&1
REM Copy just the executables (ffmpeg.exe, ffprobe.exe, ffplay.exe)
copy /Y "%FF_SOURCE_DIR%ffmpeg.exe" "%FF_DIR%\" >nul
copy /Y "%FF_SOURCE_DIR%ffprobe.exe" "%FF_DIR%\" >nul 2>&1
copy /Y "%FF_SOURCE_DIR%ffplay.exe" "%FF_DIR%\" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Failed to copy FFmpeg files to tools directory.
    goto :EOF
)

echo FFmpeg installed to %FF_DIR%.

if exist "%FF_ZIP%" del "%FF_ZIP%" >nul 2>&1
if exist "%FF_EXTRACT%" rmdir /s /q "%FF_EXTRACT%" >nul 2>&1
goto :EOF

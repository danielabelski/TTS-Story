#!/usr/bin/env bash
set -e

echo "========================================"
echo "TTS-Story Setup (Linux/macOS)"
echo "========================================"
echo
echo "IMPORTANT: Initial setup can take several minutes (large downloads + builds)."
echo "Please be patient and report any errors you encounter."
echo
echo "Quick Troubleshooting:"
echo "  - If setup fails, delete the 'venv' folder and re-run setup.sh"
echo "  - GPU users: update to the latest NVIDIA drivers"

# PyTorch versions (matching setup.bat)
TORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"
TORCHAUDIO_VERSION="2.6.0"

# 1/12 Check Python installation
echo
echo "[1/12] Checking Python installation..."
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is not installed or not in PATH"
    echo "Please install Python 3.9 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
echo "Found $PYTHON_VERSION"

# Check Python version (3.9+ required)
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo "ERROR: Python 3.9 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

# 2/12 Create virtual environment
echo
echo "[2/12] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping..."
else
    python3 -m venv venv
fi

# 3/12 Activate virtual environment
echo
echo "[3/12] Activating virtual environment..."
# shellcheck disable=SC1091
source venv/bin/activate

# 4/12 Upgrade pip
echo
echo "[4/12] Upgrading pip..."
python -m pip install --upgrade pip --quiet

# 5/12 Install PyTorch
echo
echo "[5/12] Installing PyTorch..."
echo "This may take several minutes..."
echo

# Detect NVIDIA GPU
HAS_NVIDIA=0
GPU_NAME=""
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
    if [ -" ]; then
n "$GPU_NAME        HAS_NVIDIA=1
        echo "NVIDIA GPU detected: $GPU_NAME"
    fi
fi

if [ "$HAS_NVIDIA" -eq 0 ]; then
    echo "No NVIDIA GPU detected. Using CPU-only installs."
fi

# Check if PyTorch is already installed
TORCH_INSTALLED=""
TORCH_CUDA=""
NEED_TORCH_INSTALL=1

if python -c "import torch" 2>/dev/null; then
    TORCH_INSTALLED=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
    if python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null | grep -q "cuda"; then
        TORCH_CUDA="cuda"
    else
        TORCH_CUDA="cpu"
    fi
    
    if [ "$HAS_NVIDIA" -eq 1 ] && [ "$TORCH_CUDA" = "cuda" ]; then
        echo "Detected existing CUDA torch: $TORCH_INSTALLED"
        NEED_TORCH_INSTALL=0
    elif [ "$HAS_NVIDIA" -eq 0 ] && [ "$TORCH_CUDA" = "cpu" ]; then
        echo "Detected existing CPU torch: $TORCH_INSTALLED"
        NEED_TORCH_INSTALL=0
    fi
fi

if [ "$NEED_TORCH_INSTALL" -eq 0 ]; then
    echo "Skipping torch install - compatible build already present."
else
    if [ "$HAS_NVIDIA" -eq 1 ]; then
        # Try CUDA 12.4 first (most stable), then fall back
        echo "Installing PyTorch with CUDA 12.4 support..."
        if pip install torch==${TORCH_VERSION}+cu124 torchvision==${TORCHVISION_VERSION}+cu124 torchaudio==${TORCHAUDIO_VERSION}+cu124 --index-url https://download.pytorch.org/whl/cu124 2>/dev/null; then
            echo "PyTorch CUDA 12.4 installed successfully!"
        else
            # Try CUDA 12.1
            echo "CUDA 12.4 failed, trying CUDA 12.1..."
            if pip install torch==${TORCH_VERSION}+cu121 torchvision==${TORCHVISION_VERSION}+cu121 torchaudio==${TORCHAUDIO_VERSION}+cu121 --index-url https://download.pytorch.org/whl/cu121 2>/dev/null; then
                echo "PyTorch CUDA 12.1 installed successfully!"
            else
                # Try CUDA 12.6
                echo "CUDA 12.1 failed, trying CUDA 12.6..."
                if pip install torch==${TORCH_VERSION}+cu126 torchvision==${TORCHVISION_VERSION}+cu126 torchaudio==${TORCHAUDIO_VERSION}+cu126 --index-url https://download.pytorch.org/whl/cu126 2>/dev/null; then
                    echo "PyTorch CUDA 12.6 installed successfully!"
                else
                    echo "WARNING: CUDA PyTorch install failed, trying CPU version..."
                    pip install torch torchvision torchaudio
                fi
            fi
        fi
    else
        echo "Installing CPU-only PyTorch..."
        pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
        pip install --upgrade --force-reinstall torch==${TORCH_VERSION}+cpu torchvision==${TORCHVISION_VERSION}+cpu torchaudio==${TORCHAUDIO_VERSION}+cpu --index-url https://download.pytorch.org/whl/cpu
        pip install --upgrade "numpy<1.26.0" "pillow<12.0" "fsspec<=2025.3.0" "filelock>=3.20.1,<4"
    fi
fi

# 6/12 Install other dependencies
echo
echo "[6/12] Installing other Python dependencies..."
# Filter out torch packages (already installed) and pyopenjtalk (needs special handling)
grep -vi "^torch" requirements.txt > temp_requirements.txt 2>/dev/null || true
grep -vi "^pyopenjtalk" temp_requirements.txt > temp_requirements_filtered.txt 2>/dev/null || true
pip install -r temp_requirements_filtered.txt
rm -f temp_requirements.txt temp_requirements_filtered.txt

# Install pyopenjtalk if possible (requires compile tools)
echo
echo "Checking for pyopenjtalk (Japanese text support)..."
if command -v make >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
    echo "Build tools found. Installing pyopenjtalk..."
    pip install pyopenjtalk || echo "WARNING: pyopenjtalk failed to install. Japanese TTS features will be unavailable."
else
    echo "WARNING: Build tools not found. Skipping pyopenjtalk."
    echo "To enable Japanese TTS, install build tools and rerun setup.sh."
fi

# 7/12 Install Chatterbox Turbo runtime
echo
echo "[7/12] Installing Chatterbox Turbo runtime..."
pip install chatterbox-tts --no-deps || echo "WARNING: Failed to install chatterbox-tts"

# 8/12 Install VoxCPM runtime
echo
echo "[8/12] Installing VoxCPM 1.5 runtime..."
pip install voxcpm --no-deps || echo "WARNING: Failed to install voxcpm - VoxCPM engine will not be available"

# 9/12 Install optional performance extras
echo
echo "[9/12] Installing optional performance extras..."
echo "- flash-attn (Qwen3 speedup)"
echo "- hf_xet (faster Hugging Face downloads)"

if [ "$HAS_NVIDIA" -eq 1 ]; then
    # Try to install flash-attn for Qwen3 speedup
    if pip install flash-attn --no-build-isolation 2>/dev/null; then
        echo "flash-attn installed successfully!"
    else
        echo "WARNING: flash-attn install failed. Qwen3 will use eager attention (slower)."
    fi
else
    echo "CPU-only system detected. Skipping flash-attn install."
fi

pip install hf_xet || echo "WARNING: hf_xet install failed. Hugging Face downloads may be slower."

# Ensure voice prompts directory exists
echo
echo "[10/12] Creating data directories..."
mkdir -p data/voice_prompts

# 11/12 Install system tools (espeak-ng, sox, ffmpeg)
echo
echo "[11/12] Checking system dependencies..."

# Check for apt (Debian/Ubuntu)
install_apt() {
    echo "Installing system packages via apt..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq espeak-ng sox ffmpeg libsox-dev rubberband-cli || echo "WARNING: Some system packages failed to install"
}

# Check for brew (macOS)
install_brew() {
    echo "Installing system packages via Homebrew..."
    brew install espeak-ng sox ffmpeg rubberband || echo "WARNING: Some system packages failed to install"
}

# Check for pacman (Arch Linux)
install_pacman() {
    echo "Installing system packages via pacman..."
    sudo pacman -Sy --noconfirm espeak-ng sox ffmpeg rubberband || echo "WARNING: Some system packages failed to install"
}

# Check for dnf (Fedora)
install_dnf() {
    echo "Installing system packages via dnf..."
    sudo dnf install -y espeak-ng sox ffmpeg rubberband || echo "WARNING: Some system packages failed to install"
}

if command -v apt-get >/dev/null 2>&1; then
    install_apt
elif command -v brew >/dev/null 2>&1; then
    install_brew
elif command -v pacman >/dev/null 2>&1; then
    install_pacman
elif command -v dnf >/dev/null 2>&1; then
    install_dnf
else
    echo "WARNING: Could not detect package manager. Please install manually:"
    echo "  - espeak-ng"
    echo "  - sox"
    echo "  - ffmpeg"
    echo "  - rubberband-cli"
fi

# Verify espeak-ng
echo
echo "========================================"
echo "Checking espeak-ng..."
echo "========================================"
if command -v espeak-ng >/dev/null 2>&1; then
    echo "espeak-ng is installed!"
else
    echo "WARNING: espeak-ng not found!"
    echo "Please install espeak-ng using your package manager:"
    echo "  Ubuntu/Debian: sudo apt-get install espeak-ng"
    echo "  macOS: brew install espeak-ng"
    echo "  Arch: sudo pacman -S espeak-ng"
fi

# Verify rubberband
if command -v rubberband >/dev/null 2>&1; then
    echo "rubberband is installed!"
else
    echo "WARNING: rubberband-cli not found!"
fi

# Verify ffmpeg
if command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg is installed!"
else
    echo "WARNING: ffmpeg not found!"
fi

# 12/12 Verify installation
echo
echo "[12/12] Verifying Installation..."
echo "========================================"
echo

python - << 'EOF'
import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device:", torch.cuda.get_device_name(0))
else:
    print("CUDA Device: CPU-only")
EOF

echo
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo
echo "Next steps:"
echo "  1. If espeak-ng is not installed, install it now"
echo "  2. Run: ./run.sh"
echo "  3. Open browser to: http://localhost:5000"
echo

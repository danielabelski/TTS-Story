#!/usr/bin/env bash
set -e

REPO_URL="https://github.com/Xerophayze/TTS-Story.git"
REPO_DIR="TTS-Story"

echo "========================================"
echo "TTS-Story Linux/macOS Install/Update"
echo "========================================"
echo

# Fix git safe.directory warning (allows git operations in any directory)
git config --global --add safe.directory "*" 2>/dev/null || true

# Check if git is installed
echo "Checking Git installation..."
if ! command -v git >/dev/null 2>&1; then
    echo "Git not found. Installing Git..."
    
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq git
    elif command -v brew >/dev/null 2>&1; then
        brew install git
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -Sy --noconfirm git
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y git
    elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y git
    elif command -v apk >/dev/null 2>&1; then
        sudo apk add git
    else
        echo "ERROR: Could not detect package manager to install git."
        echo "Please install git manually and re-run this script."
        exit 1
    fi
    
    # Verify git is now installed
    if ! command -v git >/dev/null 2>&1; then
        echo "ERROR: Git installation failed. Please install git manually."
        exit 1
    fi
fi

echo "Git is installed: $(git --version)"
echo

# Pre-check: Install python3-venv with ensurepip if needed
echo "Checking Python venv support..."
# Test if we can actually create a venv (not just check --help)
TEST_VENV="/tmp/venv_test_$$"
if ! python3 -m venv "$TEST_VENV" >/dev/null 2>&1; then
    echo "venv creation failed. Installing python3-venv..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3-venv python3-pip
    elif command -v brew >/dev/null 2>&1; then
        brew install python@3.10
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y python3.10-venv
    fi
fi
rm -rf "$TEST_VENV"

# Clone or update repository
echo "Cloning or updating repository..."
if [ -d "$REPO_DIR" ]; then
    if [ -d "$REPO_DIR/.git" ]; then
        echo "Repository found. Pulling latest updates..."
        # Fix permissions in case repo was cloned with sudo
        if [ -O "$REPO_DIR" ]; then
            echo "Repository owned by current user."
        else
            echo "Fixing repository ownership..."
            sudo chown -R $(whoami):$(whoami) "$REPO_DIR" 2>/dev/null || true
        fi
        cd "$REPO_DIR"
        git pull
        cd ..
    else
        echo "ERROR: $REPO_DIR exists but is not a Git repository."
        echo "Please rename or remove the folder and re-run this script."
        exit 1
    fi
else
    echo "Cloning repository..."
    git clone "$REPO_URL" "$REPO_DIR"
fi

echo
echo "========================================"
echo "Running setup.sh to install dependencies..."
echo "========================================"
echo

if [ -f "$REPO_DIR/setup.sh" ]; then
    cd "$REPO_DIR"
    chmod +x setup.sh
    
    # Remove incomplete venv if exists (caused by failed/missing python3-venv)
    if [ -d "venv" ] && [ ! -f "venv/bin/activate" ]; then
        echo "Removing incomplete virtual environment..."
        rm -rf venv
    fi
    
    ./setup.sh
    cd ..
else
    echo "ERROR: setup.sh not found in $REPO_DIR."
    exit 1
fi

echo
echo "✅ Install/update complete."

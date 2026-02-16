#!/usr/bin/env bash
set -e

REPO_URL="https://github.com/Xerophayze/TTS-Story.git"
REPO_DIR="TTS-Story"

echo "========================================"
echo "TTS-Story Linux/macOS Install/Update"
echo "========================================"
echo

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

# Clone or update repository
echo "Cloning or updating repository..."
if [ -d "$REPO_DIR" ]; then
    if [ -d "$REPO_DIR/.git" ]; then
        echo "Repository found. Pulling latest updates..."
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
    ./setup.sh
    cd ..
else
    echo "ERROR: setup.sh not found in $REPO_DIR."
    exit 1
fi

echo
echo "✅ Install/update complete."

#!/bin/bash
# Setup script for Reachy Wake Word Tester

set -e

echo "Setting up Reachy Wake Word Tester..."
echo ""

cd "$(dirname "${BASH_SOURCE[0]}")"


# Check if we're in the right directory
if [ ! -f "reachy.onnx" ]; then
    echo "Error: reachy.onnx not found!"
    echo "Please run this script from the reachy-wakeword-test directory"
    exit 1
fi

# Create virtual environment
echo "[1/4] Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "[2/4] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "[3/4] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "[4/4] Installing dependencies..."

# Install core dependencies first
pip install numpy pyaudio onnxruntime scipy scikit-learn requests tqdm pyyaml

# Install openwakeword (with dependencies to get all resource files)
pip install 'openwakeword>=0.5.0'

# Download feature extraction models
echo "[5/5] Downloading feature extraction models..."
python -c "
from openwakeword.utils import download_models
import os

# Find the openwakeword installation directory
import openwakeword
models_dir = os.path.join(os.path.dirname(openwakeword.__file__), 'resources', 'models')
os.makedirs(models_dir, exist_ok=True)

# Download all required models
print('Downloading models...')
download_models(target_directory=models_dir)
print('✓ Models downloaded successfully')
"

echo ""
echo "✓ Setup complete!"
echo ""
echo "To run the tester:"
echo "  ./run.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python test_wakeword.py"

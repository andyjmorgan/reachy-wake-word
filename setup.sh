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

# Install openwakeword without dependencies (avoids tflite conflict)
pip install --no-deps openwakeword>=0.5.0

# Copy feature extraction models from training project
echo "[5/5] Copying feature extraction models..."
MODELS_SRC="/home/localuser/source/reachy/openWakeWord/openwakeword/resources/models"
MODELS_DST="venv/lib/python3.12/site-packages/openwakeword/resources/models"

if [ -d "$MODELS_SRC" ]; then
    mkdir -p "$MODELS_DST"
    cp "$MODELS_SRC"/*.onnx "$MODELS_DST/"
    echo "✓ Models copied (melspectrogram.onnx, embedding_model.onnx)"
else
    echo "⚠ Warning: Could not find models in training project"
    echo "   You may need to copy them manually"
fi

echo ""
echo "✓ Setup complete!"
echo ""
echo "To run the tester:"
echo "  ./run.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python test_wakeword.py"

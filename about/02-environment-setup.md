# Environment Setup

## System Specifications

- **OS**: Linux (Ubuntu Noble) 6.14.0-37-generic
- **Python**: 3.12
- **GPU**: NVIDIA RTX 4090
- **CPU**: AMD (used for training due to ONNX runtime)

## Initial Repository Setup

```bash
# Clone openWakeWord
git clone https://github.com/dscripka/openWakeWord.git
cd openWakeWord

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Initial installation attempt
pip install -e .
```

## Critical Bug #1: Python 3.12 Incompatibility

**Problem**: TensorFlow 2.8.1 not compatible with Python 3.12

**Error**:
```
Could not find a version that satisfies the requirement tensorflow==2.8.1
```

**Fix**: Modified `setup.py`

```python
# Before:
'tensorflow==2.8.1'
'torchaudio>=0.13.1,<1'

# After:
'tensorflow-cpu>=2.16.0'  # Python 3.12 compatible
'torchaudio>=0.13.1,<3'   # Allow newer versions
```

**Location**: `setup.py:28-29`

## TTS Model Setup

### Piper Sample Generator

```bash
# Download Piper sample generator
git clone https://github.com/dscripka/piper-sample-generator.git

# Download TTS model (1.7 GB)
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.pt \
  -O piper-sample-generator/models/en_US-libritts_r-medium.pt

# Download room impulse response files for augmentation
cd piper-sample-generator
git clone https://github.com/Piper-Audio/piper-impulse-responses.git impulses
```

### Feature Extraction Models

```python
# Download ONNX models
import openwakeword
openwakeword.utils.download_models()
```

**Downloaded**:
- `melspectrogram.onnx` (1.09 MB) - Audio preprocessing
- `embedding_model.onnx` (1.33 MB) - Feature embedding

## Final Dependencies

**Core packages**:
```
tensorflow-cpu>=2.16.0
torch>=2.0.0
torchaudio>=0.13.1,<3
onnxruntime>=1.16.0
soundfile>=0.12.0
librosa>=0.10.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
tqdm>=4.65.0
requests>=2.28.0
```

**Audio processing**:
```
pyaudio>=0.2.13
portaudio19-dev (system package)
```

**TTS and phoneme processing**:
```
dp-phonemizer (DeepPhonemizer)
speechbrain
```

## System Dependencies

```bash
# PortAudio for microphone access
sudo apt-get install portaudio19-dev

# ALSA development files (for pyaudio)
sudo apt-get install libasound2-dev

# Audio playback tools
sudo apt-get install sox ffmpeg
```

## Directory Structure

```
openWakeWord/
├── openwakeword/          # Main package
├── models/                # Downloaded ONNX models
├── piper-sample-generator/# TTS engine
│   ├── models/           # TTS models
│   └── impulses/         # Room impulse responses
├── trained_models/        # Training output
│   └── reachy/           # Reachy model files
├── venv/                 # Python virtual environment
├── setup.py              # Modified for Python 3.12
└── reachy_config.yaml    # Training configuration
```

## Verification

After setup, verify everything works:

```bash
# Test ONNX runtime
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"

# Test audio input
python -c "import pyaudio; p = pyaudio.PyAudio(); print(f'Audio devices: {p.get_device_count()}')"

# Test TTS
python -c "from piper_sample_generator import generate_samples; print('Piper TTS ready')"

# Test openWakeWord
python -c "from openwakeword.model import Model; print('openWakeWord ready')"
```

## Notes

- **GPU**: Used for TTS generation (Piper), but CPU for ONNX inference
- **Python 3.12**: Required multiple dependency version adjustments
- **TFLite**: Skipped due to deprecated dependencies (ONNX sufficient)
- **Audio**: PulseAudio/PipeWire used for device routing (not direct ALSA)

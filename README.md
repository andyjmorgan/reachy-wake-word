# Reachy Wake Word Tester

Self-contained wake word detection testing application for Reachy robot.

## Quick Start

1. **Setup** (first time only):
   ```bash
   ./setup.sh
   ```

2. **Run**:
   ```bash
   ./run.sh
   ```

3. Say **"reachy"** and watch for detections!

## Files

- `reachy.onnx` - Trained wake word model (840 KB)
- `test_wakeword.py` - Main testing application
- `setup.sh` - One-time setup script
- `run.sh` - Run the tester
- `requirements.txt` - Python dependencies

## Features

- **Auto-detects** Reachy's microphone (no manual configuration needed)
- **Real-time score display** with visual feedback
- **Detection statistics** when stopped (Ctrl+C)
- **Threshold tuning** recommendations

## Usage

### Basic usage:
```bash
./run.sh
```

### List available devices:
```bash
./run.sh --list-devices
```

### Use specific device:
```bash
./run.sh --device 9
```

### Adjust threshold:
```bash
# Command line (temporary)
./run.sh --threshold 0.4  # More sensitive
./run.sh --threshold 0.6  # Less sensitive

# Or edit config.yaml (permanent)
# Change: threshold: 0.5
# To:     threshold: 0.4
```

## Troubleshooting

**No detections:**
- Check microphone volume in `pavucontrol`
- Try lowering threshold: `./run.sh --threshold 0.4`
- Ensure "Reachy Mini Audio" is set as default input

**Wrong microphone:**
- List devices: `./run.sh --list-devices`
- Use specific device: `./run.sh --device <number>`

**Too many false positives:**
- Raise threshold: `./run.sh --threshold 0.6`

## Performance

- **Accuracy**: 82.95%
- **Recall**: 73%
- **Model**: ONNX (840 KB)
- **Sample Rate**: 16 kHz
- **Latency**: ~80ms per chunk

## Integration

To integrate with Reachy robot control:

```python
from test_wakeword import find_reachy_microphone
from openwakeword.model import Model
import pyaudio
import numpy as np

# Find microphone
device_index, _ = find_reachy_microphone()

# Setup audio stream
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1,
                   rate=16000, input=True,
                   frames_per_buffer=1280,
                   input_device_index=device_index)

# Load model
model = Model(wakeword_models=["reachy.onnx"],
             inference_framework="onnx")

# Detection loop
while True:
    audio_data = np.frombuffer(stream.read(1280), dtype=np.int16)
    prediction = model.predict(audio_data)

    if prediction["reachy"] > 0.5:
        print("Wake word detected!")
        # Trigger your robot actions here
```

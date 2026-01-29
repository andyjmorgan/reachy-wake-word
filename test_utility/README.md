# Wake Word Test Utility

Real-time wake word detection tester with live audio visualization.

## Quick Start

```bash
# From repo root
source venv/bin/activate
python test_utility/test_wakeword.py

# Or use the run script
./test_utility/run.sh
```

## Features

- Auto-detects microphone (PulseAudio, PipeWire, or system default)
- Real-time score visualization with progress bar
- Model selection menu (lists models from `../models/`)
- Detection cooldown to prevent duplicates
- Statistics on exit (detection count, max/avg scores)

## Usage

```bash
# Interactive model selection
python test_utility/test_wakeword.py

# Specify model directly
python test_utility/test_wakeword.py --model reechy-spk150-steps100k-acc98.15-rec97.50.onnx

# Custom threshold (default: 0.5)
python test_utility/test_wakeword.py --threshold 0.4

# List audio devices
python test_utility/test_wakeword.py --list-devices

# Use specific audio device
python test_utility/test_wakeword.py --device 5
```

## Controls

| Key | Action |
|-----|--------|
| `c` | Clear console |
| `Ctrl+C` | Stop and show statistics |

## Output

```
Score: 0.892 │████████████████████████████████████████████████████░░░░░░░░│ 🔥 DETECTED!
[14:32:15] 🎤 Detection #1 | Score: 0.892
```

## Configuration

Edit `config.yaml`:

```yaml
threshold: 0.5      # Detection sensitivity (0.0-1.0)
cooldown: 2.0       # Seconds between detections
device_index: null  # null = auto-detect
chunk_size: 1280    # Audio samples per frame (80ms at 16kHz)
```

## Requirements

- Python 3.8+
- PyAudio
- openwakeword (installed from parent repo)
- Working microphone

# Reachy Wake Word

![Reachy Wake Word Training](https://cdn-uploads.huggingface.co/production/uploads/671faa3a541a76b548647676/uEa13KsL5wtQREVZ1ixwc.png)

Custom wake word model for the [Reachy robot](https://www.pollen-robotics.com/reachy/), achieving **98.15% accuracy** and **97.5% recall**.

Built on [openWakeWord](https://github.com/dscripka/openWakeWord) with systematic training experiments documented in [TRAINING_SUMMARY.md](TRAINING_SUMMARY.md).

---

## Quick Start

### Use the Pre-trained Model

```python
from openwakeword.model import Model

model = Model(
    wakeword_models=["models/reechy-spk150-steps100k-acc98.15-rec97.50.onnx"],
    inference_framework="onnx"
)

# Process 16-bit 16kHz PCM audio frames
prediction = model.predict(audio_frame)
score = prediction["reechy-spk150-steps100k-acc98.15-rec97.50"]

if score > 0.5:
    print("Wake word detected!")
```

### Test with Microphone

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Run test utility
python test_utility/test_wakeword.py
```

---

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 98.15% |
| Recall | 97.5% |
| False Positives/Hour | 0.53 |
| Wake Word | "reechy" |

---

## Training Configuration

The winning model was trained with:

```yaml
target_phrase: ["reechy"]
max_speakers: 150
n_samples: 5000
steps: 100000
layer_size: 192
tts_model_path: "en-us-libritts-high.pt"  # High quality, 904 speakers
background_paths: []  # Clean training
```

---

## Key Findings

128 experiments across 6 phases revealed:

| Finding | Impact |
|---------|--------|
| 100K training steps (vs 50K default) | +3% accuracy |
| High-quality TTS model | +1% accuracy |
| 150 speakers (not 100 or 200) | Optimal diversity |
| Clean training (no background noise) | Better validation scores |
| Single word training | Multi-word hurts by 4-9% |

See [TRAINING_SUMMARY.md](TRAINING_SUMMARY.md) for complete experiment results.

---

## Repository Structure

```
├── models/                    # Pre-trained wake word model
├── test_utility/              # Real-time testing tool
├── experiments/               # Training experiment results
├── openwakeword/              # Training framework (fork of openWakeWord)
├── TRAINING_SUMMARY.md        # Full experiment documentation
└── README.md
```

---

## Training Your Own Model

### 1. Setup Environment

```bash
git clone https://github.com/andyjmorgan/reachy-wake-word.git
cd reachy-wake-word
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Clone piper-sample-generator for TTS
git clone https://github.com/rhasspy/piper-sample-generator.git
cd piper-sample-generator && pip install -e . && cd ..

# Download required data
python -m openwakeword.download_features
wget https://github.com/dscripka/openwakeword/releases/download/v0.1.0/validation_set_features.npy

# Download TTS model
cd piper-sample-generator/models
wget https://github.com/rhasspy/piper/releases/download/2023.11.14-2/voice-en-us-libritts-high.tar.gz
tar xzf voice-en-us-libritts-high.tar.gz && cd ../..
```

### 2. Train Model

```bash
python -m openwakeword.train \
  --training_config experiments/phase4b/p4b_spk150_stp100k.yaml \
  --generate_clips --augment_clips --train_model
```

### 3. Test Model

```bash
python test_utility/test_wakeword.py --model your_model.onnx
```

---

## Acknowledgements

- [openWakeWord](https://github.com/dscripka/openWakeWord) — Core wake word detection framework
- [Piper](https://github.com/rhasspy/piper) — Text-to-speech for synthetic training data
- [Pollen Robotics](https://www.pollen-robotics.com/) — Reachy robot platform

---

## License

Code is licensed under **Apache 2.0**. Pre-trained models are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

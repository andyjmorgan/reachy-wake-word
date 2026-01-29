# Trained Models

Pre-trained wake word models for "reachy" robot.

## Best Model

**File:** `reechy-spk150-steps100k-acc98.15-rec97.50.onnx`

| Metric | Value |
|--------|-------|
| Accuracy | 98.15% |
| Recall | 97.50% |
| False Positives/Hour | 0.53 |
| Wake Word | "reechy" |

### Training Configuration

```yaml
target_phrase: ["reechy"]
model_name: "p4b_spk150_stp100k"

# TTS
tts_model_path: "piper-sample-generator/models/en-us-libritts-high.pt"
max_speakers: 150
n_samples: 5000
n_samples_val: 1000

# Training
steps: 100000
model_type: "dnn"
layer_size: 192
n_blocks: 1

# Negatives
batch_n_per_class:
  ACAV100M_sample: 1024
  adversarial_negative: 50
  positive: 50
max_negative_weight: 1500

# Augmentation
augmentation_rounds: 2
background_paths: []  # Clean training
```

### Key Training Decisions

1. **100K steps** — Doubled from default 50K, biggest accuracy gain
2. **150 speakers** — Sweet spot for HQ TTS model
3. **Layer 192** — Larger than default 128
4. **Clean training** — No background noise augmentation
5. **HQ TTS model** — `en-us-libritts-high.pt` (904 speakers)

### Usage

```python
from openwakeword.model import Model

model = Model(
    wakeword_models=["models/reechy-spk150-steps100k-acc98.15-rec97.50.onnx"],
    inference_framework="onnx"
)

# Get prediction for audio frame (16-bit 16kHz PCM)
prediction = model.predict(audio_frame)
score = prediction["reechy-spk150-steps100k-acc98.15-rec97.50"]

if score > 0.5:
    print("Wake word detected!")
```

### Test Command

```bash
python examples/detect_from_microphone.py \
  --model_path models/reechy-spk150-steps100k-acc98.15-rec97.50.onnx \
  --inference_framework onnx
```

---

See [TRAINING_SUMMARY.md](../TRAINING_SUMMARY.md) for full experiment history.

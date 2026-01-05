# Complete Configuration Reference

This document details all configuration parameters used for training the Reachy wake word model.

## Final Production Configuration

**File**: `reachy_config.yaml`

See the complete configuration file in this directory: [`reachy_config.yaml`](reachy_config.yaml)

---

## Configuration Breakdown

### Basic Settings

```yaml
target_phrase: ["reachy"]
model_name: "reachy"
```

**Explanation**:
- `target_phrase`: The wake word to train (can be multiple)
- `model_name`: Output directory name under `output_dir`

---

### TTS Settings (Piper)

```yaml
piper_sample_generator_path: "/path/to/piper-sample-generator"
tts_model_path: "/path/to/en_US-libritts_r-medium.pt"
n_samples: 5000
n_samples_val: 1000
tts_batch_size: 50
```

**Explanation**:
- `piper_sample_generator_path`: Path to Piper TTS repository
- `tts_model_path`: Pre-trained TTS model file (1.7 GB)
- `n_samples`: Number of positive training samples to generate
- `n_samples_val`: Number of validation samples
- `tts_batch_size`: GPU batch size (50 optimal for RTX 4090)

**TTS Model**: `en_US-libritts_r-medium.pt`
- Language: US English
- Quality: Medium (good balance of quality/speed)
- Size: 1.7 GB
- Speakers: Multiple (LibriTTS dataset)

---

### Training Parameters

```yaml
steps: 50000
target_accuracy: 0.7
target_recall: 0.5
target_false_positives_per_hour: 0.2
```

**Explanation**:
- `steps`: Total training steps (50K for large dataset)
- `target_accuracy`: Training target for accuracy (70%)
- `target_recall`: Training target for recall (50%)
- `target_false_positives_per_hour`: Target FP rate (0.2/hour)

**Note**: Targets are aspirational; final metrics may differ

---

### Batch Configuration

```yaml
batch_n_per_class:
  "ACAV100M_sample": 1024
  "adversarial_negative": 50
  "positive": 50

max_negative_weight: 1500
```

**Explanation**:
- `batch_n_per_class`: Number of samples per class in each batch
  - **ACAV100M_sample**: 1024 (91% of batch) - diverse background
  - **adversarial_negative**: 50 (4.5%) - phonetically similar
  - **positive**: 50 (4.5%) - wake word samples

- `max_negative_weight`: Maximum weight for negative samples
  - Gradually increases during training to reduce false positives
  - Higher = more emphasis on rejecting negatives

**Batch Weighting Strategy**:
- Heavy emphasis on general negatives (1024:50 ratio)
- Model learns "most audio is not the wake word"
- Prevents false positives on general speech/noise

---

### Model Architecture

```yaml
model_type: "dnn"
layer_size: 128
n_blocks: 1
```

**Explanation**:
- `model_type`: "dnn" (Deep Neural Network) vs "lstm"
  - DNN: Simpler, faster, good for wake words
  - LSTM: More complex, better for longer sequences

- `layer_size`: Hidden layer dimension (128)
  - Larger = more capacity but slower inference
  - 128 is good balance for wake words

- `n_blocks`: Number of hidden blocks (1)
  - More blocks = deeper network
  - 1 block sufficient for wake word detection

**Model Size**: Results in 840 KB ONNX file

---

### Audio Augmentation

```yaml
rir_paths:
  - "/path/to/impulses"
augmentation_rounds: 2
augmentation_batch_size: 100
```

**Explanation**:
- `rir_paths`: Room impulse response files for reverberation
- `augmentation_rounds`: Number of augmentation passes per clip (2)
- `augmentation_batch_size`: Batch size for augmentation processing

**Augmentations Applied**:
1. Room impulse response (convolution)
2. Pitch shifting (±2 semitones)
3. Band-stop filtering
4. Colored noise (pink/brown)
5. Gain adjustment (-6 to +6 dB)

---

### Background Noise (Intentionally Empty for Baseline)

```yaml
background_paths: []
background_paths_duplication_rate: []
```

**Explanation**:
- Baseline training had no background noise
- Production training uses pre-computed ACAV100M features instead
- These parameters would be used for raw audio background files

---

### Output Directory

```yaml
output_dir: "/path/to/trained_models"
```

**Explanation**:
- Base directory for all training outputs
- Creates subdirectory `{output_dir}/{model_name}/`
- Contains: clips, features, model files, logs

---

### Negative Training Data

```yaml
custom_negative_phrases: []
```

**Explanation**:
- Custom phrases similar to wake word
- Empty because DeepPhonemizer generates these automatically
- Could add specific phrases if needed

**Auto-generated adversarial negatives**:
- "reaching", "peachy", "teach me", "reach it", etc.
- Based on phoneme similarity to "reachy"

---

### Validation Data

```yaml
false_positive_validation_data_path: "/path/to/validation_set_features.npy"
```

**Explanation**:
- Pre-computed features for false-positive testing
- Shape: (481345, 96) - 11 hours of audio
- Sources: DiPCo, Santa Barbara Corpus, MUSDB
- Used to measure FP rate during training

**Important**: Must be 2D array (timesteps, features) for sliding window

---

### Feature Data Files

```yaml
feature_data_files:
  "ACAV100M_sample": "/path/to/acav100m_features.npy"
  "adversarial_negative": "/path/to/negative_features_train.npy"
  "positive": "/path/to/positive_features_train.npy"
```

**Explanation**:
- Maps class labels to pre-computed feature files
- Avoids recomputing features on each training run
- Enables memory-mapped loading for large datasets

**File Details**:
- `acav100m_features.npy`: 17 GB, shape (5625000, 16, 96)
- `negative_features_train.npy`: 30 MB, shape (5000, 16, 96)
- `positive_features_train.npy`: 30 MB, shape (5000, 16, 96)

---

## Training Sequence Behavior

The training runs multiple sequences with increasing negative weights:

### Sequence 1: Initial Training
- **Steps**: 50,000 (from config)
- **Negative weight**: 1.0 → max_negative_weight (1500)
- **Goal**: Learn basic wake word pattern

### Sequence 2: Negative Weight Increase
- **Steps**: 5,000 (10% of main training)
- **Negative weight**: 1500 → 1666.67 (+11%)
- **Goal**: Further reduce false positives

### Sequence 3: Final Refinement
- **Steps**: 5,000
- **Negative weight**: 1666.67 → 1851.85 (+11%)
- **Goal**: Final false-positive reduction

**Total Steps**: 60,000

---

## Feature Extraction Settings

Not in config file, but built into openWakeWord:

```python
# Mel-spectrogram parameters
sample_rate: 16000 Hz
window_size: 25 ms
hop_length: 10 ms
n_mels: 96
fmin: 20 Hz
fmax: 8000 Hz

# Output shape per clip
timesteps: 16 (1 second of audio)
features: 96 (mel bins)
```

---

## Comparison: Baseline vs Production

| Parameter | Baseline | Production | Reason |
|-----------|----------|------------|--------|
| `steps` | 10,000 | 50,000 | Larger dataset needs more training |
| `max_negative_weight` | 100 | 1500 | Stronger FP reduction |
| `batch_n_per_class` | 50:50 | 1024:50:50 | Heavily weight negatives |
| `feature_data_files` | 2 classes | 3 classes | Add ACAV100M |
| `false_positive_validation` | 16K samples | 481K samples | Proper validation set |

---

## Tips for Custom Training

### For Different Wake Words

```yaml
target_phrase: ["your_phrase"]
model_name: "your_phrase"
```

### For More Training Data

```yaml
n_samples: 10000  # More samples = better model
steps: 100000     # More steps for more data
```

### For Faster Training (Testing)

```yaml
n_samples: 1000
steps: 5000
batch_n_per_class:
  "adversarial_negative": 50
  "positive": 50
# Remove ACAV100M
```

### For Lower False Positives

```yaml
max_negative_weight: 2000  # Increase from 1500
batch_n_per_class:
  "ACAV100M_sample": 2048  # Increase from 1024
  # Keep others the same
```

### For Embedded Deployment

```yaml
model_type: "dnn"  # Simpler than LSTM
layer_size: 64     # Smaller than 128
n_blocks: 1        # Keep minimal
```

---

## Configuration Validation

Before training, verify:

```bash
# Check paths exist
ls "$piper_sample_generator_path"
ls "$tts_model_path"
ls -d "$output_dir"

# Check feature files
python -c "import numpy as np; f=np.load('acav100m_features.npy', mmap_mode='r'); print(f.shape)"

# Validate config
python -m openwakeword.train --config reachy_config.yaml --validate-only
```

---

## Common Configuration Mistakes

1. **Wrong dictionary format**:
   ```yaml
   # Wrong:
   batch_n_per_class: 50

   # Correct:
   batch_n_per_class:
     "0": 50
   ```

2. **Empty feature_data_files**:
   ```yaml
   # Wrong:
   feature_data_files: {}

   # Correct: Must populate all classes
   ```

3. **3D validation data**:
   ```python
   # Wrong: (1000, 16, 96)
   # Correct: (16000, 96) - reshaped
   ```

4. **Relative paths**:
   ```yaml
   # Wrong: "./models/tts.pt"
   # Correct: "/absolute/path/to/models/tts.pt"
   ```

---

## Final Configuration Summary

**Key Decisions**:
- ✓ DNN architecture (simple, fast)
- ✓ 50K steps (balanced training time)
- ✓ 1024:50:50 batch ratio (heavy negative weighting)
- ✓ ACAV100M features (2000 hours background)
- ✓ Proper validation set (11 hours)
- ✓ max_negative_weight: 1500 (strong FP reduction)

**Result**: 82.95% accuracy, 73% recall, 840 KB model

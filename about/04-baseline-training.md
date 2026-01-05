# Baseline Model Training

## Overview

The baseline model was trained **without background noise datasets** for quick validation of the training pipeline.

**Purpose**: Test that everything works before committing to 50,000+ training steps with large datasets

**Result**: Functional but insufficient for production use

---

## Configuration

### Training Parameters

```yaml
# File: reachy_config.yaml (baseline version)

target_phrase: ["reachy"]
model_name: "reachy"

# TTS Settings
n_samples: 5000  # Positive training samples
n_samples_val: 1000  # Validation samples
tts_batch_size: 50  # Optimized for RTX 4090

# Training Parameters (BASELINE)
steps: 10000  # Reduced for quick training
target_accuracy: 0.7
target_recall: 0.5
target_false_positives_per_hour: 0.2
batch_n_per_class:
  "0": 50  # Negative samples per batch
  "1": 50  # Positive samples per batch
max_negative_weight: 100  # Lower than production

# Model Architecture
model_type: "dnn"
layer_size: 128
n_blocks: 1

# Augmentation
rir_paths:
  - "/path/to/impulses"
augmentation_rounds: 2
augmentation_batch_size: 100

# Feature Files
feature_data_files:
  "0": "/path/to/negative_features_train.npy"
  "1": "/path/to/positive_features_train.npy"

false_positive_validation_data_path: "/path/to/fp_validation.npy"
```

---

## Training Process

### Step 1: Clip Generation (~10 minutes)

```bash
python -m openwakeword.train --config reachy_config.yaml
```

**Output**:
- 5,000 positive training clips ("reachy")
- 1,000 positive test clips
- 5,000 negative training clips (adversarial phrases)
- 1,000 negative test clips

**Phoneme Analysis**:
```
Target: "reachy" → [R][IY][CH][IY] ("ree-chee")

Similar phrases generated:
- "reaching" → [R][IY][CH][IH][NG]
- "peachy" → [P][IY][CH][IY]
- "teach me" → [T][IY][CH] [M][IY]
- "reach it" → [R][IY][CH] [IH][T]
```

### Step 2: Audio Augmentation (~2 minutes)

**Augmentations applied**:
- Room impulse response (convolution with RIR)
- Pitch shifting (±2 semitones)
- Band-stop filtering (random frequency band removal)
- Colored noise addition (pink/brown noise)
- Gain adjustments (-6 to +6 dB)

### Step 3: Feature Extraction (~3 minutes)

For each clip:
1. Compute mel-spectrogram (96 mel bins, 25ms windows)
2. Extract embedding features using pre-trained model
3. Output shape: (16 timesteps, 96 features)

**Files Generated**:
- `positive_features_train.npy` (30 MB) - 5,000 clips
- `positive_features_test.npy` (5.9 MB) - 1,000 clips
- `negative_features_train.npy` (30 MB) - 5,000 clips
- `negative_features_test.npy` (5.9 MB) - 1,000 clips

### Step 4: Model Training (~15 minutes)

**Training Sequences**:

1. **Sequence 1**: 10,000 steps (primary training)
   - Loss decreased from 0.693 → 0.512
   - Accuracy increased to 0.50

2. **Sequence 2**: 1,000 steps (negative weight → 133.33)
   - Fine-tuning with higher negative emphasis

3. **Sequence 3**: 1,000 steps (negative weight → 177.77)
   - Further refinement

**Total**: 12,000 steps in ~15 minutes

---

## Results

### Final Metrics

```
Final Model Accuracy: 0.5 (50%)
Final Model Recall: 0.0 (0%)
Final Model False Positives per Hour: 0.0
```

### Analysis

**Accuracy 50%**: Model is essentially guessing randomly (coin flip)

**Recall 0%**: Model fails to detect ANY positive samples
- This means it's too conservative
- Rejects everything, including actual wake words

**FP/hour 0.0**: No false positives because it rejects everything
- This is artificially good due to over-rejection
- Not a meaningful metric when recall is 0%

### Why Did It Fail?

**Insufficient Training Data**:
- Only 5,000 adversarial negatives
- No diverse background noise (speech, music, ambient)
- Model only learned "phonetically similar ≠ wake word"
- Didn't learn "general audio ≠ wake word"

**Imbalanced Learning**:
- 50:50 batch ratio insufficient
- Production models use 1024:50 ratio
- Model needs to see MUCH more negative data

---

## Comparison with Production Target

| Metric | Baseline | Target | Gap |
|--------|----------|--------|-----|
| Accuracy | 50% | >80% | -30% |
| Recall | 0% | >70% | -70% |
| Training Data | 10K samples | 2000 hours | 200,000x |
| Training Steps | 12K | 60K | 5x |

---

## Key Findings

✓ **Pipeline Works**: All components functional
✓ **Fast Iteration**: Training completes in ~30 minutes total
✗ **Insufficient Data**: Model cannot generalize
✗ **Production Unsuitable**: 0% recall unacceptable

---

## Next Steps

The baseline confirmed the training pipeline works but revealed the need for:

1. **Large-scale negative data** - ACAV100M (2,000 hours)
2. **Proper validation set** - 11 hours of diverse audio
3. **Longer training** - 50,000+ steps
4. **Heavier batch weighting** - 1024:50:50 ratio
5. **Higher negative weights** - Up to 1500 (from 100)

→ **Proceed to Production Training**

---

## Files Generated

**Model**:
- `trained_models/reachy.onnx` (840 KB)

**Training Data**:
- `trained_models/reachy/positive_train/` - 5,000 WAV files
- `trained_models/reachy/positive_test/` - 1,000 WAV files
- `trained_models/reachy/negative_train/` - 5,000 WAV files
- `trained_models/reachy/negative_test/` - 1,000 WAV files

**Feature Files**:
- `trained_models/reachy/positive_features_train.npy` (30 MB)
- `trained_models/reachy/negative_features_train.npy` (30 MB)
- `trained_models/reachy/fp_validation.npy` (6.4 KB)

**Logs**:
- `training_complete.log`
- `augment_all_fixes.log`

# Wake Word Training Summary

**Project:** Custom "reachy" wake word for Reachy robot
**Target:** >95% accuracy
**Hardware:** RTX 4090 (24GB)
**Duration:** Jan 25-28, 2026
**Final Result:** **98.15% accuracy, 97.5% recall**

---

## Quick Start

```bash
# Test the best model
cd /home/localuser/source/reachy-wakeword-test
./run.sh
# Select: reechy-spk150-steps100k-acc98.15-rec97.50.onnx
```

---

## Executive Summary

| Phase | Model | Word(s) | Best Acc | Best Recall | Key Finding |
|-------|-------|---------|----------|-------------|-------------|
| Baseline | — | reachy | 82.95% | 73.0% | Starting point |
| 2 | Medium | reachy | 94.50% | 89.6% | 100 speakers optimal |
| 3 | Medium | reachy | 93.85% | 88.1% | Clean > noise training |
| 3B | HQ | 4 words | 91.20% | 83.1% | Multi-word hurts (-4%) |
| 3C | HQ | reechy | 95.55% | 91.9% | HQ model is better |
| 4 | HQ | reechy | 95.20% | 91.6% | More negatives hurt |
| **4B** | **HQ** | **reechy** | **98.15%** | **97.5%** | **100K steps is key** |
| 4C | HQ | 4 words | 88.95% | 78.7% | Multi-word still hurts |
| 4D | HQ | individual | 92-96% | 86-93% | Separate models work |

**Winner:** Phase 4B — `p4b_spk150_stp100k.onnx`

---

## What We Learned

### Key Findings (Ranked by Impact)

1. **100K training steps** — Biggest single improvement (+3% over 50K)
2. **HQ TTS model** — `en-us-libritts-high.pt` beats medium model (+1%)
3. **150 speakers** — Sweet spot with HQ model (was 100 with medium)
4. **Clean training** — Noise augmentation hurts validation accuracy
5. **5K samples sufficient** — 7.5K, 10K, 15K never helped
6. **Layer 192** — Helps with HQ model (didn't with medium)
7. **Single word training** — Multi-word dilutes focus (~4-9% drop)
8. **Original negative settings** — Increasing negatives hurt performance

### What Didn't Work

| Attempt | Result | Why |
|---------|--------|-----|
| 4 wake words together | -4% to -9% | Model can't focus on multiple phonetic patterns |
| Background noise training | -5% to -15% | Validation set is clean audio |
| More samples (7.5K-15K) | No improvement | 5K provides sufficient diversity |
| More negatives (2.5x batch) | -3% | Too aggressive, hurts positive learning |
| 150K steps | -3% vs 100K | Overfitting |

---

## Repository Modifications

### Setup Fixes

1. **Python venv creation:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -e .
   ```

2. **Download required data:**
   ```bash
   # ACAV100M features (1.2GB)
   python -m openwakeword.download_features

   # Validation set
   wget https://github.com/dscripka/openwakeword/releases/download/v0.1.0/validation_set_features.npy
   ```

3. **Download TTS models:**
   ```bash
   cd piper-sample-generator/models
   # Medium (195MB, 904 speakers)
   wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json
   wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.pt

   # High quality (244MB, 904 speakers) - RECOMMENDED
   wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts/high/en-us-libritts-high.onnx.json
   wget https://github.com/rhasspy/piper/releases/download/2023.11.14-2/voice-en-us-libritts-high.tar.gz
   tar xzf voice-en-us-libritts-high.tar.gz
   ```

4. **Download background noise (optional):**
   ```bash
   mkdir -p background_noise && cd background_noise
   # ESC-50
   wget https://github.com/karoldvl/ESC-50/archive/master.zip -O esc50.zip
   unzip esc50.zip && mv ESC-50-master/audio esc50
   # FMA
   wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
   unzip fma_small.zip
   ```

### Known Issues

- **TFLite conversion fails** — `onnx-tf` version mismatch. Harmless — ONNX models work fine.
- **Short duration runs** — Cached data reuse. Delete `trained_models/<name>/` to force regeneration.

---

## Winning Configuration

```yaml
# Phase 4B Winner: 98.15% accuracy
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

# Negatives (default settings - don't increase!)
batch_n_per_class:
  ACAV100M_sample: 1024
  adversarial_negative: 50
  positive: 50
max_negative_weight: 1500

# Augmentation
augmentation_rounds: 2
background_paths: []  # Clean training works better
```

---

## All Experiment Results

### Phase 2: Speaker Diversity (Medium Model)

| Experiment | Speakers | Noise | Accuracy | Recall |
|------------|----------|-------|----------|--------|
| baseline | default | No | 85.90% | 72.70% |
| noise | default | Yes | 77.00% | 56.20% |
| 10speakers | 10 | No | 83.30% | 67.20% |
| **100speakers** | **100** | **No** | **94.50%** | **89.60%** |
| allspeakers | 904 | No | 89.65% | 80.20% |
| best_combo | 904 | Yes | 79.33% | 59.13% |

### Phase 3: Matrix Testing (Medium Model, 32 experiments)

Matrix: speakers(50,100,150,200) × noise(clean,noise) × samples(5K,7.5K) × layer(128,192)

**Top 5:**

| Speakers | Noise | Samples | Layer | Accuracy | Recall |
|----------|-------|---------|-------|----------|--------|
| 100 | clean | 5K | 128 | 93.85% | 88.1% |
| 200 | clean | 5K | 128 | 93.55% | 87.7% |
| 150 | noise | 7.5K | 192 | 93.45% | 87.6% |
| 200 | noise | 5K | 128 | 93.45% | 87.6% |
| 100 | clean | 7.5K | 128 | 92.10% | 84.5% |

### Phase 3B: HQ Model + 4 Words (32 experiments)

**Result: FAILED** — Best 91.20% (4 words diluted focus)

| Speakers | Noise | Samples | Layer | Accuracy | Recall |
|----------|-------|---------|-------|----------|--------|
| 150 | clean | 5K | 192 | 91.20% | 83.1% |
| 200 | clean | 7.5K | 192 | 89.90% | 80.5% |
| 100 | clean | 7.5K | 128 | 89.30% | 79.2% |

### Phase 3C: HQ Model + Single Word (32 experiments)

**Top 5:**

| Speakers | Noise | Samples | Layer | Accuracy | Recall |
|----------|-------|---------|-------|----------|--------|
| **150** | **clean** | **5K** | **192** | **95.55%** | **91.9%** |
| 150 | clean | 5K | 128 | 95.15% | 90.6% |
| 100 | clean | 7.5K | 192 | 93.80% | 88.3% |
| 50 | clean | 5K | 192 | 93.25% | 87.6% |
| 100 | clean | 5K | 192 | 92.25% | 84.8% |

### Phase 4: More Negatives (27 experiments)

Changed: `adversarial_negative: 128` (from 50), `max_negative_weight: 3000` (from 1500)

**Result: Worse** — Extra negatives hurt training.

| Speakers | Samples | Steps | Accuracy | Recall |
|----------|---------|-------|----------|--------|
| 175 | 5K | 100K | 95.20% | 91.6% |
| 150 | 5K | 50K | 94.20% | 89.1% |
| 150 | 5K | 100K | 93.50% | 87.6% |

### Phase 4B: Original Negatives + Longer Steps (9 experiments)

**Winner found here!**

| Speakers | Steps | Accuracy | Recall | FP/Hr |
|----------|-------|----------|--------|-------|
| **150** | **100K** | **98.15%** | **97.5%** | **0.53** |
| 175 | 100K | 98.00% | 96.5% | 1.86 |
| 125 | 100K | 97.05% | 94.8% | 0.09 |
| 175 | 150K | 95.15% | 92.4% | 1.33 |
| 150 | 75K | 80.50% | 61.6% | 0.09 |

### Phase 4C: Winning Config + 4 Words (1 experiment)

**Result: FAILED** — 88.95% accuracy, 78.7% recall. Multi-word still doesn't work.

### Phase 4D: Individual Models Per Word (4 experiments)

| Word | Accuracy | Recall | FP/Hr |
|------|----------|--------|-------|
| reechy | 95.70% | 92.8% | 0.62 |
| reeshy | 95.30% | 90.9% | 1.15 |
| reachy | 95.20% | 92.5% | 1.42 |
| rishy | 92.40% | 86.2% | 2.48 |

---

## Final Models

Location: `/home/localuser/source/reachy-wakeword-test/`

| File | Word | Accuracy | Recall | Notes |
|------|------|----------|--------|-------|
| `reechy-spk150-steps100k-acc98.15-rec97.50.onnx` | reechy | 98.15% | 97.5% | **BEST** |
| `reechy-spk150-steps100k-acc95.70-rec92.80.onnx` | reechy | 95.70% | 92.8% | Phase 4D |
| `reachy-spk150-steps100k-acc95.20-rec92.50.onnx` | reachy | 95.20% | 92.5% | Phase 4D |
| `reeshy-spk150-steps100k-acc95.30-rec90.90.onnx` | reeshy | 95.30% | 90.9% | Phase 4D |
| `rishy-spk150-steps100k-acc92.40-rec86.20.onnx` | rishy | 92.40% | 86.2% | Phase 4D |

---

## Training Command

```bash
cd /home/localuser/source/reachy/openWakeWord
source venv/bin/activate

python -m openwakeword.train \
  --training_config experiments/phase4b/p4b_spk150_stp100k.yaml \
  --generate_clips --augment_clips --train_model
```

---

## Test Command

```bash
cd /home/localuser/source/reachy-wakeword-test
./run.sh
# Select model number when prompted
# Say "reachy" / "reechy" to test
# Ctrl+C to stop
```

---

*Generated: January 28, 2026*

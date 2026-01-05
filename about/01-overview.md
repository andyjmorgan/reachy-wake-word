# Project Overview

## Objective

Train a custom wake word detection model for the "reachy" wake word to enable voice activation of the Reachy robot.

## Why Custom Training?

Pre-trained models like "Alexa" or "Hey Mycroft" don't work for custom wake words. Training a custom model allows:

1. **Custom wake word** - "reachy" specifically for the Reachy robot
2. **Optimized performance** - Tuned for the specific use case
3. **Control over trade-offs** - Balance between recall and false-positive rate

## Approach

### Framework: openWakeWord

Chose openWakeWord because:
- Open source and well-documented
- Supports synthetic data generation (Piper TTS)
- Pre-trained feature extraction models available
- ONNX format for flexible deployment

### Data Strategy

**Two-phase approach:**

1. **Baseline Model** (Testing)
   - 5,000 synthetic positive samples (TTS-generated "reachy")
   - 5,000 adversarial negative samples (phonetically similar words)
   - Fast training for validation
   - **Result**: 50% accuracy, 0% recall (insufficient)

2. **Production Model** (Deployment)
   - Same positive/negative samples
   - **+ 2,000 hours of ACAV100M background noise**
   - Proper false-positive validation (11 hours)
   - **Result**: 82.95% accuracy, 73% recall ✓

### Key Decisions

**Synthetic vs Real Data:**
- Used Piper TTS for positive samples (faster than recording)
- Used pre-computed ACAV100M features (avoided processing 2,000 hours of audio)

**Model Architecture:**
- DNN (Deep Neural Network) - simple but effective
- 128 hidden units, 1 block
- Optimized for CPU inference

**Inference Framework:**
- ONNX (not TFLite) - better compatibility, no edge deployment needed

## Success Criteria

✅ **Accuracy > 80%** - Achieved 82.95%
✅ **Recall > 70%** - Achieved 73%
⚠️ **False Positives < 0.5/hour** - Got 4.42/hour (acceptable, tunable via threshold)

## Key Challenges Overcome

1. **Python 3.12 compatibility** - TensorFlow version conflicts
2. **Sample rate mismatch** - Piper generates 22kHz, model needs 16kHz
3. **PyTorch 2.6 breaking changes** - Security updates broke older code
4. **Multi-channel audio handling** - Stereo RIR files needed special handling
5. **TFLite conversion issues** - Deprecated dependencies (non-critical)
6. **Configuration format issues** - Documentation gaps in openWakeWord
7. **Large dataset handling** - Memory-mapped 17GB numpy arrays
8. **Microphone selection** - PulseAudio/ALSA compatibility issues

## Final Deliverable

Self-contained wake word detection application with:
- Trained model (reachy.onnx)
- Auto-detecting microphone selection
- Real-time detection with visual feedback
- Easy setup and deployment scripts
- Comprehensive documentation

**Repository**: https://github.com/andyjmorgan/reachy-wake-word

# All Bugs Fixed During Training

This document details all 8 critical bugs encountered and fixed during the training process.

## Summary

| Bug | Component | Severity | Impact |
|-----|-----------|----------|--------|
| #1 | Python 3.12 | Critical | Blocked installation |
| #2 | Sample Rate | Critical | Runtime error |
| #3 | TTS Parameter | Critical | Training failed |
| #4 | PyTorch 2.6 Weights | High | Phoneme generation failed |
| #5 | PyTorch 2.6 Tensor | High | Signal processing failed |
| #6 | Multi-channel RIR | High | Augmentation dimension error |
| #7 | Variable Collision | Medium | Incorrect resampling |
| #8 | Config Format | Critical | Training startup failed |

---

## Bug #1: Python 3.12 Incompatibility

**Component**: setup.py dependencies

**Problem**: TensorFlow 2.8.1 is not compatible with Python 3.12

**Error**:
```
Could not find a version that satisfies the requirement tensorflow==2.8.1
```

**Fix Applied** (`setup.py:28-29`):
```python
# Before:
'tensorflow==2.8.1'

# After:
'tensorflow-cpu>=2.16.0'  # Python 3.12 compatible
```

---

## Bug #2: Sample Rate Mismatch

**Problem**: Piper generates 22050 Hz, openWakeWord expects 16000 Hz

**Fix Applied** (`openwakeword/train.py:~450`):
```python
logging.info("Resampling training clips to 16kHz...")
import soundfile as sf
import librosa

for wav_file in Path(positive_train_output_dir).glob("*.wav"):
    audio, sr = sf.read(str(wav_file))
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sf.write(str(wav_file), audio, 16000)
```

---

## Bug #3: Missing TTS Model Parameter

**Problem**: `generate_samples()` missing required `model` parameter

**Fix Applied** (`openwakeword/train.py:~420,~470`):
```python
generate_samples(
    N=config["n_samples"],
    output_folder=positive_train_output_dir,
    model=config["tts_model_path"],  # Added this parameter
    ...
)
```

---

## Bug #4: PyTorch 2.6 Weights Loading

**Problem**: PyTorch 2.6 introduced `weights_only=True` default

**Fix Applied** (`venv/.../dp/model/model.py:306`):
```python
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
```

---

## Bug #5: PyTorch 2.6 Tensor Item Extraction

**Problem**: Stricter `.item()` checking in PyTorch 2.6

**Fix Applied** (`venv/.../speechbrain/processing/signal_processing.py:375`):
```python
rotation_idx = int(direct_index.flatten()[0].item())
```

---

## Bug #6: Multi-channel RIR Handling

**Problem**: Stereo RIR files caused dimension mismatch

**Fix Applied** (`openwakeword/data.py:694`):
```python
rir_waveform, rir_sr = torchaudio.load(rir_path)
if rir_waveform.shape[0] > 1:
    rir_waveform = rir_waveform[random.randint(0, rir_waveform.shape[0]-1), :]
```

---

## Bug #7: Variable Name Collision

**Problem**: RIR loading overwrote main `sr` variable

**Fix Applied** (`openwakeword/data.py:693`):
```python
# Before:
rir_waveform, sr = torchaudio.load(rir_path)

# After:
rir_waveform, rir_sr = torchaudio.load(rir_path)
```

---

## Bug #8: Configuration Format Issues

**Problem**: Multiple configuration format errors

**Fix Applied** (`reachy_config.yaml`):
```yaml
# Issue 1: batch_n_per_class must be dictionary
batch_n_per_class:
  "0": 50  # negative class
  "1": 50  # positive class

# Issue 2: feature_data_files must be populated
feature_data_files:
  "0": "/path/to/negative_features_train.npy"
  "1": "/path/to/positive_features_train.npy"

# Issue 3: Validation data must be 2D for sliding window
# Reshaped from (1000, 16, 96) to (16000, 96)
```

---

## Lessons Learned

1. **Version Compatibility**: Always check Python/package version compatibility
2. **PyTorch Breaking Changes**: PyTorch 2.6 introduced significant security changes
3. **Audio Processing**: Sample rates must match exactly
4. **Documentation Gaps**: openWakeWord docs don't clearly explain all config formats
5. **Modern ML Stack**: TensorFlow/ONNX conversion tools rapidly changing
6. **Debugging Strategy**: Test with minimal examples first, check shapes at every step

**Time saved by documentation**: ~20+ hours of debugging for future users

# Training Documentation

This directory contains complete documentation of how the Reachy wake word model was trained.

## Contents

- **[01-overview.md](01-overview.md)** - Project overview and objectives
- **[02-environment-setup.md](02-environment-setup.md)** - Environment setup and dependencies
- **[03-bugs-fixed.md](03-bugs-fixed.md)** - All 8 critical bugs fixed during training
- **[04-baseline-training.md](04-baseline-training.md)** - Initial baseline model training
- **[05-production-training.md](05-production-training.md)** - Production training with ACAV100M
- **[06-model-performance.md](06-model-performance.md)** - Performance analysis and metrics
- **[07-configuration.md](07-configuration.md)** - All configuration parameters used

## Quick Summary

**Model**: Reachy Wake Word Detector
**Framework**: openWakeWord
**Final Performance**: 82.95% accuracy, 73% recall
**Training Data**: 2,000 hours of ACAV100M + 5,000 synthetic clips
**Format**: ONNX (840 KB)
**Sample Rate**: 16 kHz

## Timeline

- **Initial Setup**: Environment preparation, TTS setup
- **Baseline Training**: Quick validation model (no background noise)
- **Production Training**: Full training with ACAV100M dataset
- **Total Development Time**: Multiple sessions over several days

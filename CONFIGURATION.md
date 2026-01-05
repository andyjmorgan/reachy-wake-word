# Configuration Guide

## Quick Configuration

Edit `config.yaml` to set your preferences:

```yaml
# Detection threshold (0.0 - 1.0)
threshold: 0.5

# Cooldown between detections (seconds)
cooldown: 2.0

# Audio device (null = auto-detect)
device_index: null

# Chunk size (samples)
chunk_size: 1280
```

## Configuration Methods

### Method 1: config.yaml (Permanent)

Edit the `config.yaml` file:

```bash
nano config.yaml
```

Changes persist across runs.

### Method 2: Command Line (Temporary)

Override config for single run:

```bash
./run.sh --threshold 0.4
./run.sh --device 9
```

Command line takes priority over config file.

### Method 3: Custom Config File

Use a different config file:

```bash
./run.sh --config my_custom_config.yaml
```

## Configuration Parameters

### threshold

**Type**: Float (0.0 - 1.0)
**Default**: 0.5

Controls detection sensitivity:

| Value | Behavior | Use Case |
|-------|----------|----------|
| 0.3-0.4 | Very sensitive | Quiet room, want to catch everything |
| 0.4-0.5 | Sensitive | Normal use, prefer fewer misses |
| 0.5-0.6 | Balanced | **Recommended starting point** |
| 0.6-0.7 | Conservative | Noisy environment, prefer fewer false positives |
| 0.7-0.8 | Very conservative | Very noisy, willing to repeat wake word |

**Examples**:

```yaml
# Catch more wake words (but more false positives)
threshold: 0.4

# Fewer false positives (but might miss some)
threshold: 0.6
```

### cooldown

**Type**: Float (seconds)
**Default**: 2.0

Time between detections to prevent multiple triggers:

| Value | Behavior |
|-------|----------|
| 1.0 | Quick response, may double-trigger |
| 2.0 | **Recommended** - good balance |
| 3.0+ | Slower, but very safe |

**Examples**:

```yaml
# Faster response
cooldown: 1.5

# Prevent all double-triggers
cooldown: 3.0
```

### device_index

**Type**: Integer or null
**Default**: null (auto-detect)

Specific audio device to use:

```yaml
# Auto-detect (finds pulse/pipewire)
device_index: null

# Use specific device
device_index: 9
```

To find device numbers:

```bash
./run.sh --list-devices
```

### chunk_size

**Type**: Integer (samples)
**Default**: 1280 (80ms at 16kHz)

Audio processing chunk size:

| Value | Latency | CPU Usage |
|-------|---------|-----------|
| 640 | 40ms | Higher |
| 1280 | **80ms** (recommended) | Normal |
| 1920 | 120ms | Lower |

Most users should keep default.

## Example Configurations

### For Quiet Home Environment

```yaml
threshold: 0.45  # Slightly more sensitive
cooldown: 2.0
device_index: null
chunk_size: 1280
```

### For Noisy Office

```yaml
threshold: 0.6   # Less sensitive
cooldown: 2.5    # Longer cooldown
device_index: null
chunk_size: 1280
```

### For Demo/Testing

```yaml
threshold: 0.4   # Very sensitive
cooldown: 1.5    # Quick response
device_index: null
chunk_size: 1280
```

### For Specific Microphone

```yaml
threshold: 0.5
cooldown: 2.0
device_index: 9  # Your specific device
chunk_size: 1280
```

## Tuning Tips

### Too Many False Positives?

1. **Increase threshold**: Try 0.55, 0.6, 0.65
2. **Increase cooldown**: Try 2.5 or 3.0
3. **Check microphone volume**: Might be too sensitive

### Missing Wake Words?

1. **Lower threshold**: Try 0.45, 0.4
2. **Check microphone volume**: Might be too quiet
3. **Speak more clearly**: Emphasize "REE-chee"
4. **Move closer**: Within 2m of microphone

### How to Find Best Threshold

Run this test:

```bash
# Try different thresholds
./run.sh --threshold 0.3  # Say "reachy" 10 times, count detections
./run.sh --threshold 0.4  # Say "reachy" 10 times, count detections
./run.sh --threshold 0.5  # Say "reachy" 10 times, count detections
./run.sh --threshold 0.6  # Say "reachy" 10 times, count detections
```

**Goal**:
- Detect 7-8 out of 10 attempts
- <1 false positive per 5 minutes

## Command Line Reference

```bash
# Show help
./run.sh --help

# List audio devices
./run.sh --list-devices

# Set threshold
./run.sh --threshold 0.6

# Use specific device
./run.sh --device 9

# Combine options
./run.sh --threshold 0.5 --device 9

# Use custom config
./run.sh --config production.yaml
```

## Priority Order

When the same parameter is set multiple places:

1. **Command line** (highest priority)
2. **Custom config file** (--config)
3. **config.yaml** (default file)
4. **Built-in defaults** (lowest priority)

Example:

```bash
# config.yaml has threshold: 0.5
# Command line overrides to 0.6
./run.sh --threshold 0.6
# Uses: 0.6
```

## Resetting to Defaults

To restore default settings:

```bash
# Restore default config.yaml
git checkout config.yaml

# Or manually set:
threshold: 0.5
cooldown: 2.0
device_index: null
chunk_size: 1280
```

## Advanced: Environment-Specific Configs

Create multiple config files:

```bash
# config_quiet.yaml
threshold: 0.45
cooldown: 2.0

# config_noisy.yaml
threshold: 0.6
cooldown: 2.5

# Use them:
./run.sh --config config_quiet.yaml
./run.sh --config config_noisy.yaml
```

#!/usr/bin/env python3
"""
Reachy Wake Word Tester - Auto-detecting version
Automatically finds Reachy's microphone and tests wake word detection
"""

import pyaudio
import numpy as np
from openwakeword.model import Model
import time
import sys
import os
import select
import yaml
import termios
import tty

def find_reachy_microphone():
    """Auto-detect Reachy's microphone or use PulseAudio"""
    audio = pyaudio.PyAudio()

    # Priority order for device selection
    search_terms = [
        ("pulse", "PulseAudio"),
        ("pipewire", "PipeWire"),
        ("default", "System Default"),
        ("reachy mini audio", "Reachy Mini Audio"),
        ("reachy", "Reachy Device")
    ]

    print("Searching for Reachy's microphone...")

    for search_term, description in search_terms:
        for i in range(audio.get_device_count()):
            try:
                info = audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    if search_term in info['name'].lower():
                        print(f"✓ Found: {description}")
                        print(f"  Device #{i}: {info['name']}")
                        print(f"  Channels: {info['maxInputChannels']}")
                        print(f"  Sample Rate: {info['defaultSampleRate']} Hz\n")
                        audio.terminate()
                        return i, info
            except:
                continue

    # Fallback to first available input device
    print("⚠ Could not find preferred device, using first available input...")
    for i in range(audio.get_device_count()):
        try:
            info = audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  Device #{i}: {info['name']}\n")
                audio.terminate()
                return i, info
        except:
            continue

    audio.terminate()
    print("❌ No input devices found!")
    sys.exit(1)

def select_model():
    """List available .onnx models and prompt user to select one"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "..", "models")

    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        sys.exit(1)

    models = sorted([f for f in os.listdir(models_dir) if f.endswith('.onnx')])

    if not models:
        print("❌ No .onnx models found in ../models/")
        sys.exit(1)

    print("\nAvailable Models:")
    print("=" * 80)
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print("=" * 80)

    while True:
        try:
            choice = input(f"\nSelect model [1-{len(models)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected = models[idx]
                print(f"✓ Selected: {selected}\n")
                return os.path.join(models_dir, selected)
            print(f"Please enter a number between 1 and {len(models)}")
        except (ValueError, EOFError):
            print(f"Please enter a number between 1 and {len(models)}")

def test_wake_word(device_index, threshold=0.5, model_path=None):
    """Run wake word detection"""

    if model_path is None:
        model_path = select_model()

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    model_filename = os.path.basename(model_path)
    wake_key = os.path.splitext(model_filename)[0]
    model_name = model_filename

    # Initialize audio
    audio = pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1280

    try:
        mic_stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=device_index
        )
    except Exception as e:
        print(f"❌ Failed to open microphone: {e}")
        print("\nTry running with a specific device:")
        print("  python test_wakeword.py --device <number>")
        audio.terminate()
        sys.exit(1)

    # Load model
    print(f"Loading {model_name} wake word model...")
    try:
        owwModel = Model(wakeword_models=[model_path], inference_framework="onnx")
        print(f"✓ {model_name} model loaded!\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        mic_stream.close()
        audio.terminate()
        sys.exit(1)

    # Display info
    print("="*80)
    print(f"REACHY WAKE WORD DETECTOR - {model_name.upper()} MODEL")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Threshold: {threshold}")
    print(f"Wake Word: 'reachy'")
    print("\nLegend:")
    print("  Score < 0.3  : Background/silence")
    print("  Score 0.3-0.4: Possibly similar word")
    print(f"  Score > {threshold}  : DETECTION! 🔥")
    print("="*80)
    print("\nSay 'reachy' to test...")
    print("Press Ctrl+C to stop | Press 'c' to clear console\n")

    # Detection loop
    detection_count = 0
    last_detection = 0
    cooldown = 2.0
    max_score_seen = 0.0
    scores_buffer = []

    # Set terminal to raw mode for single-char input
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            # Check for keyboard input (non-blocking)
            if select.select([sys.stdin], [], [], 0.0)[0]:
                key = sys.stdin.read(1)
                if key.lower() == 'c':
                    print("\033[2J\033[H")  # Clear screen + home
                    print("Console cleared - listening...\n")

            # Get audio
            audio_data = np.frombuffer(
                mic_stream.read(CHUNK, exception_on_overflow=False),
                dtype=np.int16
            )

            # Get prediction
            prediction = owwModel.predict(audio_data)
            score = prediction[wake_key]

            # Track scores
            scores_buffer.append(score)
            if len(scores_buffer) > 100:
                scores_buffer.pop(0)

            max_score_seen = max(max_score_seen, score)

            # Visual score bar
            bar_length = int(min(score, 1.0) * 60)
            bar = "█" * bar_length + "░" * (60 - bar_length)

            # Status indicator
            if score > threshold:
                status = "🔥 DETECTED!"
            elif score > 0.4:
                status = "⚠️  CLOSE   "
            elif score > 0.3:
                status = "👀 MAYBE   "
            else:
                status = "   ...     "

            # Print live score
            print(f"\rScore: {score:.3f} │{bar}│ {status}", end='', flush=True)

            # Detection with cooldown
            current_time = time.time()
            if score > threshold and (current_time - last_detection) > cooldown:
                detection_count += 1
                last_detection = current_time

                # One-line compact detection
                timestamp = time.strftime('%H:%M:%S')
                print(f"\n[{timestamp}] 🎤 Detection #{detection_count} | Score: {score:.3f}")
                print(f"\rScore: {score:.3f} │{bar}│ {status}", end='', flush=True)

    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("STOPPED")
        print("="*80)
        print(f"Detections: {detection_count} | Threshold: {threshold} | Max Score: {max_score_seen:.3f}")

        if scores_buffer:
            avg = np.mean(scores_buffer)
            print(f"Avg: {avg:.3f} | Range: {np.min(scores_buffer):.3f}-{np.max(scores_buffer):.3f}")

        # Compact recommendations
        if max_score_seen < threshold and max_score_seen > 0:
            suggested = max(0.3, max_score_seen * 0.9)
            print(f"\n💡 Try: ./run.sh --threshold {suggested:.2f}")
        elif detection_count == 0:
            print(f"\n💡 No detections - check mic volume")
        elif detection_count >= 5:
            print(f"\n✓ Working well!")

        print("="*80)

    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        mic_stream.stop_stream()
        mic_stream.close()
        audio.terminate()

def load_config(config_path=None):
    """Load configuration from config.yaml"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                print(f"✓ Loaded config from: {config_path}")
                return config or {}
        except Exception as e:
            print(f"Warning: Could not load {config_path}: {e}")
    return {}

if __name__ == "__main__":
    import argparse

    # Load config file first
    config = load_config()

    parser = argparse.ArgumentParser(description="Reachy Wake Word Tester")
    parser.add_argument("--device", type=int, default=config.get('device_index'),
                       help="Specific device index to use (default: auto-detect or from config)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Detection threshold (default: 0.5)")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to .onnx model file (if not set, shows selection menu)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to custom config file")
    parser.add_argument("--list-devices", action="store_true",
                       help="List all available audio devices")

    args = parser.parse_args()

    # Load custom config if specified
    if args.config:
        config = load_config(args.config)
        if not args.device and 'device_index' in config:
            args.device = config['device_index']
        # Threshold from command line takes priority, otherwise use config
        if args.threshold is None and 'threshold' in config:
            args.threshold = config['threshold']

    # List devices if requested
    if args.list_devices:
        audio = pyaudio.PyAudio()
        print("\nAvailable Input Devices:")
        print("="*80)
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"Device #{i}: {info['name']}")
                print(f"  Channels: {info['maxInputChannels']}")
                print(f"  Sample Rate: {info['defaultSampleRate']} Hz\n")
        audio.terminate()
        sys.exit(0)

    # Auto-detect or use specified device
    if args.device is not None:
        device_index = args.device
        print(f"Using specified device #{device_index}\n")
    else:
        device_index, _ = find_reachy_microphone()

    # Resolve model path
    model_path = None
    if args.model:
        if os.path.exists(args.model):
            model_path = args.model
        else:
            # Try in ../models/
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "..", "models", args.model)
            if not os.path.exists(model_path):
                model_path = os.path.join(script_dir, args.model)

    # Run test
    test_wake_word(device_index, args.threshold, model_path)

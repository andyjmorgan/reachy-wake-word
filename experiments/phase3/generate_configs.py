#!/usr/bin/env python3
"""Generate Phase 3 experiment configs for comprehensive matrix test."""

import itertools
import yaml
from pathlib import Path

# Matrix variables
SPEAKERS = [50, 100, 150, 200]
NOISE = [False, True]
SAMPLES = [5000, 7500]
LAYER_SIZE = [128, 192]

# Fixed values
BASE_CONFIG = {
    "target_phrase": ["reachy"],
    "piper_sample_generator_path": "/home/localuser/source/reachy/openWakeWord/piper-sample-generator",
    "tts_model_path": "/home/localuser/source/reachy/openWakeWord/piper-sample-generator/models/en_US-libritts_r-medium.pt",
    "tts_batch_size": 50,
    "steps": 50000,
    "target_accuracy": 0.7,
    "target_recall": 0.5,
    "target_false_positives_per_hour": 0.2,
    "batch_n_per_class": {
        "ACAV100M_sample": 1024,
        "adversarial_negative": 50,
        "positive": 50
    },
    "max_negative_weight": 1500,
    "model_type": "dnn",
    "n_blocks": 1,
    "rir_paths": ["/home/localuser/source/reachy/openWakeWord/piper-sample-generator/impulses"],
    "augmentation_rounds": 2,
    "augmentation_batch_size": 100,
    "output_dir": "/home/localuser/source/reachy/openWakeWord/trained_models",
    "custom_negative_phrases": [],
    "false_positive_validation_data_path": "/home/localuser/source/reachy/openWakeWord/validation_set_features.npy",
}

BACKGROUND_PATHS = [
    "/home/localuser/source/reachy/openWakeWord/background_noise/esc50",
    "/home/localuser/source/reachy/openWakeWord/background_noise/fma_small"
]

output_dir = Path(__file__).parent
experiments = []

for speakers, noise, samples, layer_size in itertools.product(SPEAKERS, NOISE, SAMPLES, LAYER_SIZE):
    # Create experiment name
    noise_str = "noise" if noise else "clean"
    name = f"p3_spk{speakers}_{noise_str}_smp{samples}_lay{layer_size}"

    config = BASE_CONFIG.copy()
    config["model_name"] = name
    config["max_speakers"] = speakers
    config["n_samples"] = samples
    config["n_samples_val"] = samples // 5  # 20% for validation
    config["layer_size"] = layer_size

    if noise:
        config["background_paths"] = BACKGROUND_PATHS
        config["background_paths_duplication_rate"] = [1, 1]
    else:
        config["background_paths"] = []
        config["background_paths_duplication_rate"] = []

    # Feature data files (model-specific paths)
    config["feature_data_files"] = {
        "ACAV100M_sample": "/home/localuser/source/reachy/openWakeWord/acav100m_features.npy",
        "adversarial_negative": f"/home/localuser/source/reachy/openWakeWord/trained_models/{name}/negative_features_train.npy",
        "positive": f"/home/localuser/source/reachy/openWakeWord/trained_models/{name}/positive_features_train.npy"
    }

    # Write config file
    config_path = output_dir / f"{name}.yaml"
    with open(config_path, 'w') as f:
        f.write(f"# Phase 3: speakers={speakers}, noise={noise}, samples={samples}, layer_size={layer_size}\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    experiments.append({
        "name": name,
        "speakers": speakers,
        "noise": noise_str,
        "samples": samples,
        "layer_size": layer_size
    })
    print(f"Created: {name}.yaml")

# Write experiment list
with open(output_dir / "experiment_list.txt", 'w') as f:
    for exp in experiments:
        f.write(f"{exp['name']}\n")

print(f"\nGenerated {len(experiments)} experiment configs")

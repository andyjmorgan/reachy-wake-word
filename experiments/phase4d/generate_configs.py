#!/usr/bin/env python3
"""Phase 4D: Individual models per word with winning config."""

import yaml
from pathlib import Path

WORDS = ["reachy", "reeshy", "rishy", "reechy"]

BASE_CONFIG = {
    "piper_sample_generator_path": "/home/localuser/source/reachy/openWakeWord/piper-sample-generator",
    "tts_model_path": "/home/localuser/source/reachy/openWakeWord/piper-sample-generator/models/en-us-libritts-high.pt",
    "tts_batch_size": 50,
    "n_samples": 5000,
    "n_samples_val": 1000,
    "steps": 100000,
    "max_speakers": 150,
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
    "layer_size": 192,
    "n_blocks": 1,
    "rir_paths": ["/home/localuser/source/reachy/openWakeWord/piper-sample-generator/impulses"],
    "augmentation_rounds": 2,
    "augmentation_batch_size": 100,
    "output_dir": "/home/localuser/source/reachy/openWakeWord/trained_models",
    "custom_negative_phrases": [],
    "false_positive_validation_data_path": "/home/localuser/source/reachy/openWakeWord/validation_set_features.npy",
    "background_paths": [],
    "background_paths_duplication_rate": [],
}

output_dir = Path(__file__).parent

for word in WORDS:
    name = f"p4d_{word}"
    config = BASE_CONFIG.copy()
    config["target_phrase"] = [word]
    config["model_name"] = name

    config["feature_data_files"] = {
        "ACAV100M_sample": "/home/localuser/source/reachy/openWakeWord/acav100m_features.npy",
        "adversarial_negative": f"/home/localuser/source/reachy/openWakeWord/trained_models/{name}/negative_features_train.npy",
        "positive": f"/home/localuser/source/reachy/openWakeWord/trained_models/{name}/positive_features_train.npy"
    }

    config_path = output_dir / f"{name}.yaml"
    with open(config_path, 'w') as f:
        f.write(f"# Phase 4D: '{word}' | 150 spk, 100K steps, lay192, HQ\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Created: {name}.yaml")

print(f"\nGenerated {len(WORDS)} configs (one per word)")

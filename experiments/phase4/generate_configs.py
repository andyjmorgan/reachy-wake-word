#!/usr/bin/env python3
"""Phase 4: Zero in on sweet spot. Scale negatives, steps, samples around winner."""

import itertools
import yaml
from pathlib import Path

# Fine-tune around winner: 150 spk, clean, 5K, lay192, HQ model
SPEAKERS = [125, 150, 175]
SAMPLES = [5000, 10000, 15000]
STEPS = [50000, 75000, 100000]
# Fixed: clean, layer 192, HQ model

# Increased negative training
BASE_CONFIG = {
    "target_phrase": ["reechy"],
    "piper_sample_generator_path": "/home/localuser/source/reachy/openWakeWord/piper-sample-generator",
    "tts_model_path": "/home/localuser/source/reachy/openWakeWord/piper-sample-generator/models/en-us-libritts-high.pt",
    "tts_batch_size": 50,
    "target_accuracy": 0.7,
    "target_recall": 0.5,
    "target_false_positives_per_hour": 0.2,
    "batch_n_per_class": {
        "ACAV100M_sample": 1024,
        "adversarial_negative": 128,  # Up from 50
        "positive": 50
    },
    "max_negative_weight": 3000,  # Up from 1500
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
experiments = []

for speakers, samples, steps in itertools.product(SPEAKERS, SAMPLES, STEPS):
    name = f"p4_spk{speakers}_smp{samples}_stp{steps//1000}k"

    config = BASE_CONFIG.copy()
    config["model_name"] = name
    config["max_speakers"] = speakers
    config["n_samples"] = samples
    config["n_samples_val"] = samples // 5
    config["steps"] = steps

    config["feature_data_files"] = {
        "ACAV100M_sample": "/home/localuser/source/reachy/openWakeWord/acav100m_features.npy",
        "adversarial_negative": f"/home/localuser/source/reachy/openWakeWord/trained_models/{name}/negative_features_train.npy",
        "positive": f"/home/localuser/source/reachy/openWakeWord/trained_models/{name}/positive_features_train.npy"
    }

    config_path = output_dir / f"{name}.yaml"
    with open(config_path, 'w') as f:
        f.write(f"# Phase 4: spk={speakers}, smp={samples}, steps={steps//1000}k | lay192, clean, HQ, more negatives\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    experiments.append(name)
    print(f"Created: {name}.yaml")

with open(output_dir / "experiment_list.txt", 'w') as f:
    for exp in experiments:
        f.write(f"{exp}\n")

print(f"\nGenerated {len(experiments)} experiments")
print(f"Changes from 3C: adversarial_negative batch 50→128, max_negative_weight 1500→3000")

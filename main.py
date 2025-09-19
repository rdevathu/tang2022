#!/usr/bin/env python3
"""
Main script for ECG AF recurrence prediction following Tang et al. 2022.

This script implements the complete pipeline:
1. Load CARTO dataset
2. Run 10-fold stratified cross-validation
3. Train ECG-only CNN models
4. Evaluate with patient-level aggregation
5. Generate comprehensive results

Usage:
    python main.py [--config config.json]
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.tang2022.carto_data_interface import load_carto_dataset
from src.tang2022.training import run_cross_validation


def load_config(config_path: str = None) -> dict:
    """Load configuration from JSON file or use defaults."""
    default_config = {
        "data_dir": "data",
        "results_dir": "results",
        "n_folds": 10,
        "n_epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "random_seed": 42,
        "model_size": "base",
        "patience": 15,
    }

    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            user_config = json.load(f)
        default_config.update(user_config)
        print(f"Loaded configuration from {config_path}")
    else:
        print("Using default configuration")

    return default_config


def create_default_config(output_path: str = "config.json"):
    """Create a default configuration file."""
    config = {
        "data_dir": "data",
        "results_dir": "results",
        "n_folds": 10,
        "n_epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "random_seed": 42,
        "model_size": "base",
        "patience": 15,
        "description": "Configuration for Tang et al. 2022 ECG AF recurrence prediction",
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created default configuration file: {output_path}")
    return config


def print_system_info():
    """Print system and environment information."""
    import numpy as np
    import pandas as pd
    import sklearn
    import torch

    print("System Information:")
    print(f"  Python: {sys.version}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  NumPy: {np.__version__}")
    print(f"  Pandas: {pd.__version__}")
    print(f"  Scikit-learn: {sklearn.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA devices: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name()}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="ECG AF Recurrence Prediction (Tang et al. 2022)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to configuration JSON file",
    )

    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create default configuration file and exit",
    )

    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run quick test with 2 folds and 3 epochs",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing CARTO dataset",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    # Create config and exit if requested
    if args.create_config:
        create_default_config()
        return

    # Print header
    print("=" * 60)
    print("ECG AF Recurrence Prediction (Tang et al. 2022)")
    print("=" * 60)

    # Print system info
    print_system_info()
    print()

    # Load configuration
    config = load_config(args.config)

    # Override with command line arguments
    if args.data_dir != "data":
        config["data_dir"] = args.data_dir
    if args.results_dir != "results":
        config["results_dir"] = args.results_dir

    # Test run configuration
    if args.test_run:
        print("Running in TEST MODE (quick run)")
        config.update(
            {
                "n_folds": 2,
                "n_epochs": 3,
                "batch_size": 16,
                "results_dir": "test_results",
            }
        )

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    try:
        # Load CARTO dataset
        print("Loading CARTO dataset...")
        carto_dataset = load_carto_dataset(config["data_dir"])

        # Print dataset statistics
        stats = carto_dataset.get_dataset_statistics()
        print("\nDataset Statistics:")
        print(f"  Total patients: {stats['unique_patients']}")
        print(f"  Label distribution: {stats['label_distribution']}")
        print(f"  Recurrence rate: {stats['recurrence_rate']:.3f}")
        print(f"  ECG shape: {stats['ecg_shape']}")
        print(f"  Sampling rate: {stats['sampling_rate']} Hz")
        print()

        # Run cross-validation
        print("Starting cross-validation...")
        results_df = run_cross_validation(
            carto_dataset=carto_dataset,
            n_folds=config["n_folds"],
            n_epochs=config["n_epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            results_dir=config["results_dir"],
            random_seed=config["random_seed"],
        )

        # Save final configuration
        config_save_path = Path(config["results_dir"]) / "final_config.json"
        with open(config_save_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nResults saved to: {config['results_dir']}")
        print("Files created:")
        print("  - cv_results.csv: Detailed fold results")
        print("  - cv_summary.json: Summary statistics")
        print("  - fold_X_predictions.json: Predictions per fold")
        print("  - fold_X_best_model.pth: Best model per fold")
        print("  - final_config.json: Final configuration used")

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Experiment failed. Check the error message above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

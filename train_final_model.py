#!/usr/bin/env python3
"""
Final model training script for ECG AF recurrence prediction.

This script implements a proper train-test split methodology to build
the best possible final model for deployment, replacing cross-validation
with a single optimized training run.

Key improvements over cross-validation approach:
1. Proper train/validation/test splits with stratification
2. Single final model optimized for best performance
3. Comprehensive evaluation on held-out test set
4. Model ready for deployment

Usage:
    python train_final_model.py [--config config_final_model.json]

    # Quick test run
    python train_final_model.py --test-run

    # Custom parameters
    python train_final_model.py --test-size 0.15 --val-size 0.15 --epochs 200
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.tang2022.carto_data_interface import load_carto_dataset
from src.tang2022.training import train_final_model


def load_config(config_path: str = None) -> dict:
    """Load configuration from JSON file or use defaults."""
    default_config = {
        "data_dir": "data",
        "results_dir": "final_model_results",
        "test_size": 0.2,
        "val_size": 0.2,
        "n_epochs": 150,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "random_seed": 42,
        "model_size": "base",
        "patience": 20,
        "training_mode": "final_model",
    }

    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            user_config = json.load(f)
        default_config.update(user_config)
        print(f"Loaded configuration from {config_path}")
    else:
        print("Using default configuration for final model training")

    return default_config


def create_final_model_config(output_path: str = "config_final_model.json"):
    """Create a configuration file optimized for final model training."""
    config = {
        "data_dir": "data",
        "results_dir": "final_model_results",
        "test_size": 0.2,
        "val_size": 0.2,
        "n_epochs": 150,
        "batch_size": 32,
        "learning_rate": 0.001,
        "random_seed": 42,
        "model_size": "base",
        "patience": 20,
        "training_mode": "final_model",
        "description": "Optimized configuration for training final ECG AF recurrence prediction model",
        "strategy": {
            "approach": "train_test_split",
            "rationale": "Build single best model for deployment",
            "validation": "holdout_test_set",
            "early_stopping": "validation_auroc",
        },
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created final model configuration file: {output_path}")
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


def validate_config(config: dict) -> dict:
    """Validate and adjust configuration for final model training."""
    # Ensure reasonable split sizes
    if config["test_size"] + config["val_size"] >= 0.8:
        print("WARNING: Test + validation size is very large. Adjusting...")
        config["test_size"] = min(0.2, config["test_size"])
        config["val_size"] = min(0.2, config["val_size"])

    # Ensure sufficient epochs for convergence
    if config["n_epochs"] < 50:
        print(
            "WARNING: Low epoch count for final model. Consider using at least 100 epochs."
        )

    # Ensure reasonable patience for early stopping
    if config["patience"] < 10:
        print(
            "WARNING: Low patience may cause premature stopping. Consider patience >= 15."
        )

    return config


def main():
    """Main execution function for final model training."""
    parser = argparse.ArgumentParser(
        description="Train Final ECG AF Recurrence Prediction Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config_final_model.json",
        help="Path to configuration JSON file",
    )

    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create default configuration file for final model training and exit",
    )

    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run quick test with reduced epochs and small test size",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing CARTO dataset",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save final model and results",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Fraction of data to hold out for final testing (0.1-0.3 recommended)",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=None,
        help="Fraction of training data to use for validation (0.15-0.25 recommended)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Maximum number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Adam optimizer learning rate",
    )

    args = parser.parse_args()

    # Create config and exit if requested
    if args.create_config:
        create_final_model_config()
        return

    # Print header
    print("=" * 70)
    print("ECG AF RECURRENCE PREDICTION - FINAL MODEL TRAINING")
    print("=" * 70)
    print("Training methodology: Single optimized model with train-test split")
    print("Objective: Build best possible model for deployment")
    print("=" * 70)

    # Print system info
    print_system_info()
    print()

    # Load configuration
    config = load_config(args.config if not args.test_run else None)

    # Override with command line arguments
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir
    if args.results_dir is not None:
        config["results_dir"] = args.results_dir
    if args.test_size is not None:
        config["test_size"] = args.test_size
    if args.val_size is not None:
        config["val_size"] = args.val_size
    if args.epochs is not None:
        config["n_epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate

    # Test run configuration
    if args.test_run:
        print("Running in TEST MODE (quick validation run)")
        config.update(
            {
                "n_epochs": 10,
                "batch_size": 16,
                "results_dir": "test_final_model_results",
                "patience": 5,
                "test_size": 0.3,  # Larger for faster testing
                "val_size": 0.3,
            }
        )

    # Validate configuration
    config = validate_config(config)

    print("\nFinal Model Training Configuration:")
    print(f"  Data directory: {config['data_dir']}")
    print(f"  Results directory: {config['results_dir']}")
    print(f"  Test set size: {config['test_size']:.1%}")
    print(f"  Validation set size: {config['val_size']:.1%}")
    print(
        f"  Training set size: ~{1 - config['test_size'] - config['val_size']:.1%}"
    )
    print(f"  Max epochs: {config['n_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Early stopping patience: {config['patience']}")
    print(f"  Random seed: {config['random_seed']}")
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

        # Calculate expected split sizes
        n_patients = stats["unique_patients"]
        n_test = int(n_patients * config["test_size"])
        n_val = int((n_patients - n_test) * config["val_size"])
        n_train = n_patients - n_test - n_val

        print("\nExpected data splits:")
        print(f"  Training: {n_train} patients")
        print(f"  Validation: {n_val} patients")
        print(f"  Test: {n_test} patients")
        print()

        # Train final model
        print("Starting final model training...")
        print("This may take significant time for optimal convergence.")
        print("Monitor validation AUROC for training progress.")
        print("-" * 50)

        final_results = train_final_model(
            carto_dataset=carto_dataset,
            test_size=config["test_size"],
            val_size=config["val_size"],
            n_epochs=config["n_epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            results_dir=config["results_dir"],
            random_seed=config["random_seed"],
        )

        # Save final configuration used
        config_save_path = (
            Path(config["results_dir"]) / "final_training_config.json"
        )
        config["actual_results"] = {
            "test_auroc": final_results["test_performance"]["auroc"],
            "test_f1": final_results["test_performance"]["f1"],
            "training_epochs": final_results["training"]["total_epochs"],
            "best_epoch": final_results["training"]["best_epoch"],
        }

        with open(config_save_path, "w") as f:
            json.dump(config, f, indent=2)

        # Print final summary
        print(f"\n{'=' * 70}")
        print("FINAL MODEL TRAINING COMPLETED SUCCESSFULLY")
        print(f"{'=' * 70}")
        print(f"Results directory: {config['results_dir']}")
        print("Model file: final_model.pth")
        print("Results file: final_model_results.json")
        print("\nModel Performance on Held-out Test Set:")
        print(f"  AUROC: {final_results['test_performance']['auroc']:.4f}")
        print(
            f"  Sensitivity: {final_results['test_performance']['sensitivity']:.4f}"
        )
        print(
            f"  Specificity: {final_results['test_performance']['specificity']:.4f}"
        )
        print(f"  F1 Score: {final_results['test_performance']['f1']:.4f}")
        print(
            f"  Accuracy: {final_results['test_performance']['accuracy']:.4f}"
        )

        print("\nTraining Summary:")
        print(f"  Total epochs: {final_results['training']['total_epochs']}")
        print(f"  Best epoch: {final_results['training']['best_epoch']}")
        print(
            f"  Best val AUROC: {final_results['training']['best_val_auroc']:.4f}"
        )

        print("\nData Splits Used:")
        print(
            f"  Training: {final_results['data_splits']['train_patients']} patients"
        )
        print(
            f"  Validation: {final_results['data_splits']['val_patients']} patients"
        )
        print(
            f"  Test: {final_results['data_splits']['test_patients']} patients"
        )

        print(f"\n{'=' * 70}")
        print("Your final model is ready for deployment!")
        print("Use 'final_model.pth' for inference on new ECG data.")
        print(f"{'=' * 70}")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Final model training failed. Check the error message above.")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

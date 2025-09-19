"""
Training and evaluation framework for ECG AF recurrence prediction.

Implements:
- 10-fold stratified cross-validation
- Training loop with Adam optimizer
- Patient-level evaluation with aggregated predictions
- Metrics calculation (AUROC, sensitivity, specificity, F1, etc.)
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from .augmentation import create_augmentation_pipeline
from .carto_data_interface import CARTODataset
from .dataset import aggregate_patient_predictions, create_dataloaders
from .model import create_ecg_model


class ECGTrainer:
    """
    Trainer for ECG AF recurrence prediction following Tang et al. 2022.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        """
        Initialize trainer.

        Args:
            model: ECG CNN model
            device: PyTorch device
            learning_rate: Adam learning rate
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()

        # Training history
        self.history = {"train_loss": [], "val_loss": [], "val_auroc": []}

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for windows, labels, patient_ids in tqdm(train_loader, desc="Training"):
            windows = windows.to(self.device)
            labels = labels.float().to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(windows)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def evaluate_epoch(
        self, test_loader: DataLoader, return_predictions: bool = False
    ) -> Dict:
        """
        Evaluate model on test data.

        Args:
            test_loader: Test data loader
            return_predictions: Whether to return detailed predictions

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        # Collect predictions by patient
        patient_predictions = {}  # patient_id -> list of probabilities
        patient_labels = {}  # patient_id -> label

        with torch.no_grad():
            for windows, labels, patient_ids in tqdm(
                test_loader, desc="Evaluating"
            ):
                windows = windows.to(self.device)
                labels_tensor = labels.float().to(self.device)

                # Forward pass
                logits = self.model(windows)
                loss = self.criterion(logits, labels_tensor)
                probabilities = torch.sigmoid(logits)

                total_loss += loss.item()
                n_batches += 1

                # Collect predictions by patient
                for i, patient_id in enumerate(patient_ids):
                    if patient_id not in patient_predictions:
                        patient_predictions[patient_id] = []
                        patient_labels[patient_id] = labels[i].item()

                    patient_predictions[patient_id].append(
                        probabilities[i].cpu().item()
                    )

        # Aggregate to patient level
        patient_probs = aggregate_patient_predictions(patient_predictions)

        # Get patient-level labels and predictions
        patients = list(patient_probs.keys())
        y_true = np.array([patient_labels[pid] for pid in patients])
        y_prob = np.array([patient_probs[pid] for pid in patients])

        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_prob)
        metrics["loss"] = total_loss / n_batches

        if return_predictions:
            metrics["predictions"] = {
                "patient_ids": patients,
                "y_true": y_true,
                "y_prob": y_prob,
            }

        return metrics

    def calculate_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: True labels [n_patients]
            y_prob: Predicted probabilities [n_patients]

        Returns:
            Dictionary with metrics
        """
        # AUROC
        try:
            auroc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auroc = 0.5  # Only one class present

        # Find optimal threshold (maximize F1)
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = (
            thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        )

        # Binary predictions at optimal threshold
        y_pred = (y_prob >= optimal_threshold).astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision_opt = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_opt = sensitivity
        f1_opt = (
            2 * (precision_opt * recall_opt) / (precision_opt + recall_opt)
            if (precision_opt + recall_opt) > 0
            else 0.0
        )

        # Brier score
        brier_score = np.mean((y_prob - y_true) ** 2)

        # Expected Calibration Error (ECE)
        ece = self.calculate_ece(y_true, y_prob, n_bins=10)

        metrics = {
            "auroc": auroc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "accuracy": accuracy,
            "precision": precision_opt,
            "recall": recall_opt,
            "f1": f1_opt,
            "brier_score": brier_score,
            "ece": ece,
            "optimal_threshold": optimal_threshold,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }

        return metrics

    def calculate_ece(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration

        Returns:
            Expected Calibration Error
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        total_samples = len(y_true)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += (
                    np.abs(avg_confidence_in_bin - accuracy_in_bin)
                    * prop_in_bin
                )

        return ece

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 50,
        patience: int = 10,
        save_path: Optional[str] = None,
    ) -> Dict:
        """
        Train model with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Maximum number of epochs
            patience: Early stopping patience
            save_path: Path to save best model

        Returns:
            Training history and best metrics
        """
        best_auroc = 0.0
        patience_counter = 0
        best_metrics = None

        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")

            # Training
            train_loss = self.train_epoch(train_loader)

            # Validation
            val_metrics = self.evaluate_epoch(val_loader)

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_auroc"].append(val_metrics["auroc"])

            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val AUROC: {val_metrics['auroc']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")

            # Early stopping
            if val_metrics["auroc"] > best_auroc:
                best_auroc = val_metrics["auroc"]
                best_metrics = val_metrics.copy()
                patience_counter = 0

                # Save best model
                if save_path:
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "epoch": epoch,
                            "best_auroc": best_auroc,
                            "metrics": best_metrics,
                        },
                        save_path,
                    )
            else:
                patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        return {
            "history": self.history,
            "best_metrics": best_metrics,
            "best_epoch": epoch - patience_counter,
        }


def run_cross_validation(
    carto_dataset: CARTODataset,
    n_folds: int = 10,
    n_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    results_dir: str = "results",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Run 10-fold stratified cross-validation following Tang et al. 2022.

    Args:
        carto_dataset: CARTO dataset instance
        n_folds: Number of CV folds
        n_epochs: Maximum epochs per fold
        batch_size: Training batch size
        learning_rate: Adam learning rate
        results_dir: Directory to save results
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with fold results
    """
    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Create results directory
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)

    # Get patient data for stratification
    patient_ids = carto_dataset.get_patient_ids()
    labels = carto_dataset.get_labels()

    print(f"Running {n_folds}-fold CV with {len(patient_ids)} patients")
    print(f"Label distribution: {np.bincount(labels)}")

    # Stratified K-fold split
    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=random_seed
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(patient_ids, labels)
    ):
        print(f"\n{'=' * 50}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'=' * 50}")

        # Split patients
        train_patients = [patient_ids[i] for i in train_idx]
        test_patients = [patient_ids[i] for i in test_idx]

        print(f"Train patients: {len(train_patients)}")
        print(f"Test patients: {len(test_patients)}")

        # Create model (8 independent channels per Tang et al. 2022)
        model = create_ecg_model("base", in_channels=8)

        # Create data loaders
        augmentation = create_augmentation_pipeline(training=True)
        train_loader, test_loader = create_dataloaders(
            carto_dataset,
            train_patients,
            test_patients,
            batch_size=batch_size,
            augmentation=augmentation,
            num_workers=0,
        )

        # Create trainer
        trainer = ECGTrainer(model, device, learning_rate=learning_rate)

        # Train model
        fold_save_path = results_path / f"fold_{fold + 1}_best_model.pth"
        training_results = trainer.train(
            train_loader,
            test_loader,
            n_epochs=n_epochs,
            patience=10,
            save_path=str(fold_save_path),
        )

        # Final evaluation
        final_metrics = trainer.evaluate_epoch(
            test_loader, return_predictions=True
        )

        # Save fold results
        fold_result = {
            "fold": fold + 1,
            "n_train_patients": len(train_patients),
            "n_test_patients": len(test_patients),
            "best_epoch": training_results["best_epoch"],
            **final_metrics,
        }

        # Remove predictions for summary (too large)
        fold_result_summary = {
            k: v for k, v in fold_result.items() if k != "predictions"
        }
        fold_results.append(fold_result_summary)

        # Save detailed predictions
        predictions_path = results_path / f"fold_{fold + 1}_predictions.json"
        predictions_data = {
            "fold": fold + 1,
            "patient_ids": final_metrics["predictions"]["patient_ids"],
            "y_true": final_metrics["predictions"]["y_true"].tolist(),
            "y_prob": final_metrics["predictions"]["y_prob"].tolist(),
        }

        with open(predictions_path, "w") as f:
            json.dump(predictions_data, f, indent=2)

        print(f"\nFold {fold + 1} Results:")
        print(f"  AUROC: {final_metrics['auroc']:.4f}")
        print(f"  Sensitivity: {final_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {final_metrics['specificity']:.4f}")
        print(f"  F1: {final_metrics['f1']:.4f}")

    # Create results DataFrame
    results_df = pd.DataFrame(fold_results)

    # Calculate summary statistics
    metric_cols = [
        "auroc",
        "sensitivity",
        "specificity",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "brier_score",
        "ece",
    ]

    summary_stats = {}
    for metric in metric_cols:
        values = results_df[metric].values
        summary_stats[f"{metric}_mean"] = np.mean(values)
        summary_stats[f"{metric}_std"] = np.std(values)

    # Save results
    results_df.to_csv(results_path / "cv_results.csv", index=False)

    with open(results_path / "cv_summary.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    # Print final summary
    print(f"\n{'=' * 50}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'=' * 50}")

    for metric in ["auroc", "sensitivity", "specificity", "f1"]:
        mean_val = summary_stats[f"{metric}_mean"]
        std_val = summary_stats[f"{metric}_std"]
        print(f"{metric.upper()}: {mean_val:.3f} ± {std_val:.3f}")

    return results_df


def train_final_model(
    carto_dataset: CARTODataset,
    test_size: float = 0.2,
    val_size: float = 0.2,
    n_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    results_dir: str = "results",
    random_seed: int = 42,
) -> Dict:
    """
    Train a single final model using proper train/validation/test split.

    This approach replaces cross-validation with a single train-test split
    to build the best possible final model for deployment.

    Args:
        carto_dataset: CARTO dataset instance
        test_size: Fraction of data to hold out for final testing
        val_size: Fraction of training data to use for validation
        n_epochs: Maximum epochs for training
        batch_size: Training batch size
        learning_rate: Adam learning rate
        results_dir: Directory to save results
        random_seed: Random seed for reproducibility

    Returns:
        Dict with training results and metrics
    """
    from sklearn.model_selection import train_test_split

    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Create results directory
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)

    # Get patient data for stratification
    patient_ids = carto_dataset.get_patient_ids()
    labels = carto_dataset.get_labels()

    print(f"Training final model with {len(patient_ids)} patients")
    print(f"Label distribution: {np.bincount(labels)}")
    print(f"Recurrence rate: {np.mean(labels):.3f}")

    # First split: separate test set
    train_val_patients, test_patients, train_val_labels, test_labels = (
        train_test_split(
            patient_ids,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=random_seed,
        )
    )

    # Second split: separate validation from training
    train_patients, val_patients, train_labels, val_labels = train_test_split(
        train_val_patients,
        train_val_labels,
        test_size=val_size,
        stratify=train_val_labels,
        random_state=random_seed,
    )

    print("\nData splits:")
    print(
        f"  Train: {len(train_patients)} patients ({np.mean(train_labels):.3f} recurrence rate)"
    )
    print(
        f"  Val:   {len(val_patients)} patients ({np.mean(val_labels):.3f} recurrence rate)"
    )
    print(
        f"  Test:  {len(test_patients)} patients ({np.mean(test_labels):.3f} recurrence rate)"
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model (8 independent channels per Tang et al. 2022)
    model = create_ecg_model("base", in_channels=8)
    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Create data loaders
    augmentation = create_augmentation_pipeline(training=True)

    train_loader, val_loader = create_dataloaders(
        carto_dataset,
        train_patients,
        val_patients,
        batch_size=batch_size,
        augmentation=augmentation,
        num_workers=0,
    )

    _, test_loader = create_dataloaders(
        carto_dataset,
        train_patients,
        test_patients,  # Use train_patients as dummy for consistency
        batch_size=batch_size,
        augmentation=None,
        num_workers=0,
    )

    # Actually create proper test loader
    test_loader, _ = create_dataloaders(
        carto_dataset,
        test_patients,
        train_patients,  # Swap to get proper test loader
        batch_size=batch_size,
        augmentation=None,
        num_workers=0,
    )

    print("\nDataloader sizes:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader)} batches")

    # Create trainer
    trainer = ECGTrainer(model, device, learning_rate=learning_rate)

    # Train model with validation for early stopping
    print("\nStarting training...")
    model_save_path = results_path / "final_model.pth"
    training_results = trainer.train(
        train_loader,
        val_loader,
        n_epochs=n_epochs,
        patience=15,
        save_path=str(model_save_path),
    )

    print("Training completed!")
    print(f"Best epoch: {training_results['best_epoch']}")
    print(
        f"Best validation AUROC: {training_results['best_metrics']['auroc']:.4f}"
    )

    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Final evaluation on test set
    print("Evaluating on held-out test set...")
    test_metrics = trainer.evaluate_epoch(test_loader, return_predictions=True)

    # Compile final results
    final_results = {
        "model_info": {
            "model_size": "base",
            "input_channels": 8,
            "parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        },
        "data_splits": {
            "train_patients": len(train_patients),
            "val_patients": len(val_patients),
            "test_patients": len(test_patients),
            "train_recurrence_rate": float(np.mean(train_labels)),
            "val_recurrence_rate": float(np.mean(val_labels)),
            "test_recurrence_rate": float(np.mean(test_labels)),
        },
        "training": {
            "total_epochs": len(training_results["history"]["train_loss"]),
            "best_epoch": training_results["best_epoch"],
            "best_val_auroc": training_results["best_metrics"]["auroc"],
            "final_train_loss": training_results["history"]["train_loss"][-1],
            "final_val_loss": training_results["history"]["val_loss"][-1],
        },
        "test_performance": {
            "auroc": float(test_metrics["auroc"]),
            "sensitivity": float(test_metrics["sensitivity"]),
            "specificity": float(test_metrics["specificity"]),
            "accuracy": float(test_metrics["accuracy"]),
            "f1": float(test_metrics["f1"]),
            "precision": float(test_metrics["precision"]),
            "recall": float(test_metrics["recall"]),
            "brier_score": float(test_metrics["brier_score"]),
            "ece": float(test_metrics["ece"]),
        },
    }

    # Save detailed results
    results_file = results_path / "final_model_results.json"
    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2)

    # Save test predictions
    predictions_file = results_path / "final_test_predictions.json"
    predictions_data = {
        "patient_ids": test_metrics["predictions"]["patient_ids"],
        "y_true": test_metrics["predictions"]["y_true"].tolist(),
        "y_prob": test_metrics["predictions"]["y_prob"].tolist(),
        "test_metrics": final_results["test_performance"],
    }

    with open(predictions_file, "w") as f:
        json.dump(predictions_data, f, indent=2)

    # Save training history
    training_history = {
        "train_losses": training_results["history"]["train_loss"],
        "val_losses": training_results["history"]["val_loss"],
        "val_aurocs": training_results["history"]["val_auroc"],
        "epochs": list(
            range(1, len(training_results["history"]["train_loss"]) + 1)
        ),
    }

    history_file = results_path / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(training_history, f, indent=2)

    # Print final summary
    print(f"\n{'=' * 60}")
    print("FINAL MODEL TRAINING COMPLETED")
    print(f"{'=' * 60}")
    print(f"Model saved to: {model_save_path}")
    print(f"Results saved to: {results_file}")
    print("\nFinal Test Performance:")
    print(f"  AUROC: {test_metrics['auroc']:.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Brier Score: {test_metrics['brier_score']:.4f}")
    print(f"  ECE: {test_metrics['ece']:.4f}")

    return final_results


if __name__ == "__main__":
    # Test training framework
    from .carto_data_interface import load_carto_dataset

    print("Testing training framework...")

    # Load dataset
    carto_dataset = load_carto_dataset()

    # Run small test (2 folds, few epochs)
    results_df = run_cross_validation(
        carto_dataset,
        n_folds=2,
        n_epochs=3,
        batch_size=16,
        results_dir="test_results",
    )

    print("Training framework test completed!")

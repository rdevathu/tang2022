#!/usr/bin/env python3
"""
Analysis and visualization script for final ECG AF recurrence prediction model.

This script provides comprehensive analysis of the final model results including:
1. Performance metrics visualization
2. Training history plots
3. Calibration analysis
4. ROC and PR curves
5. Confusion matrix
6. Patient-level prediction analysis

Usage:
    python analyze_final_model.py [--results-dir final_model_results]
    python analyze_final_model.py --results-dir test_final_model_results --save-plots
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class FinalModelAnalyzer:
    """Analyzer for final ECG model results."""

    def __init__(self, results_dir: str):
        """
        Initialize analyzer with results directory.

        Args:
            results_dir: Directory containing final model results
        """
        self.results_dir = Path(results_dir)
        self.results = None
        self.predictions = None
        self.training_history = None
        self.config = None

        # Load all result files
        self._load_results()

    def _load_results(self):
        """Load all result files from the results directory."""
        # Load main results
        results_file = self.results_dir / "final_model_results.json"
        if results_file.exists():
            with open(results_file, "r") as f:
                self.results = json.load(f)
        else:
            raise FileNotFoundError(f"Results file not found: {results_file}")

        # Load predictions
        predictions_file = self.results_dir / "final_test_predictions.json"
        if predictions_file.exists():
            with open(predictions_file, "r") as f:
                self.predictions = json.load(f)

        # Load training history
        history_file = self.results_dir / "training_history.json"
        if history_file.exists():
            with open(history_file, "r") as f:
                self.training_history = json.load(f)

        # Load configuration
        config_file = self.results_dir / "final_training_config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                self.config = json.load(f)

        print(f"Loaded results from: {self.results_dir}")

    def print_summary(self):
        """Print comprehensive summary of results."""
        print("=" * 70)
        print("FINAL ECG AF RECURRENCE PREDICTION MODEL - ANALYSIS SUMMARY")
        print("=" * 70)

        if self.results:
            # Model information
            model_info = self.results["model_info"]
            print("\nModel Architecture:")
            print(f"  Type: {model_info['model_size']} ECGNet")
            print(f"  Input channels: {model_info['input_channels']}")
            print(f"  Parameters: {model_info['parameters']:,}")

            # Data splits
            splits = self.results["data_splits"]
            print("\nData Splits:")
            print(
                f"  Training: {splits['train_patients']} patients ({splits['train_recurrence_rate']:.3f} recurrence)"
            )
            print(
                f"  Validation: {splits['val_patients']} patients ({splits['val_recurrence_rate']:.3f} recurrence)"
            )
            print(
                f"  Test: {splits['test_patients']} patients ({splits['test_recurrence_rate']:.3f} recurrence)"
            )

            # Training summary
            training = self.results["training"]
            print("\nTraining Summary:")
            print(f"  Total epochs: {training['total_epochs']}")
            print(f"  Best epoch: {training['best_epoch']}")
            print(f"  Best validation AUROC: {training['best_val_auroc']:.4f}")
            print(f"  Final train loss: {training['final_train_loss']:.4f}")
            print(f"  Final val loss: {training['final_val_loss']:.4f}")

            # Test performance
            perf = self.results["test_performance"]
            print("\nTest Set Performance:")
            print(f"  AUROC: {perf['auroc']:.4f}")
            print(f"  Sensitivity: {perf['sensitivity']:.4f}")
            print(f"  Specificity: {perf['specificity']:.4f}")
            print(f"  F1 Score: {perf['f1']:.4f}")
            print(f"  Accuracy: {perf['accuracy']:.4f}")
            print(f"  Precision: {perf['precision']:.4f}")
            print(f"  Brier Score: {perf['brier_score']:.4f}")
            print(f"  ECE: {perf['ece']:.4f}")

        print("=" * 70)

    def plot_training_history(self, save_plot: bool = False) -> plt.Figure:
        """Plot training history."""
        if not self.training_history:
            print("No training history available for plotting.")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "Final Model Training History", fontsize=16, fontweight="bold"
        )

        epochs = self.training_history["epochs"]
        train_losses = self.training_history["train_losses"]
        val_losses = self.training_history["val_losses"]
        val_aurocs = self.training_history["val_aurocs"]

        # Training and validation loss
        axes[0, 0].plot(
            epochs, train_losses, label="Training Loss", linewidth=2
        )
        axes[0, 0].plot(
            epochs, val_losses, label="Validation Loss", linewidth=2
        )
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Validation AUROC
        axes[0, 1].plot(
            epochs,
            val_aurocs,
            label="Validation AUROC",
            color="green",
            linewidth=2,
        )
        if self.results:
            best_epoch = self.results["training"]["best_epoch"]
            best_auroc = self.results["training"]["best_val_auroc"]
            axes[0, 1].axvline(
                x=best_epoch,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Best Epoch ({best_epoch})",
            )
            axes[0, 1].axhline(
                y=best_auroc, color="red", linestyle="--", alpha=0.7
            )
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("AUROC")
        axes[0, 1].set_title("Validation AUROC Progress")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Loss difference (overfitting indicator)
        loss_diff = np.array(val_losses) - np.array(train_losses)
        axes[1, 0].plot(
            epochs,
            loss_diff,
            label="Val Loss - Train Loss",
            color="orange",
            linewidth=2,
        )
        axes[1, 0].axhline(y=0, color="black", linestyle="-", alpha=0.3)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss Difference")
        axes[1, 0].set_title("Overfitting Monitor")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Learning rate (if available)
        if (
            "learning_rates" in self.training_history
            and self.training_history["learning_rates"]
        ):
            learning_rates = self.training_history["learning_rates"]
            axes[1, 1].plot(
                epochs,
                learning_rates,
                label="Learning Rate",
                color="purple",
                linewidth=2,
            )
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Learning Rate")
            axes[1, 1].set_title("Learning Rate Schedule")
            axes[1, 1].set_yscale("log")
        else:
            # Show training summary text instead
            axes[1, 1].text(
                0.1,
                0.5,
                "Training Summary:\n\n"
                + f"Total Epochs: {len(epochs)}\n"
                + f"Final Train Loss: {train_losses[-1]:.4f}\n"
                + f"Final Val Loss: {val_losses[-1]:.4f}\n"
                + f"Final Val AUROC: {val_aurocs[-1]:.4f}\n"
                + f"Best Val AUROC: {max(val_aurocs):.4f}",
                transform=axes[1, 1].transAxes,
                fontsize=12,
                verticalalignment="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )
            axes[1, 1].set_title("Training Summary")
            axes[1, 1].axis("off")

        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plot:
            plot_path = self.results_dir / "training_history.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Training history plot saved to: {plot_path}")

        return fig

    def plot_performance_metrics(self, save_plot: bool = False) -> plt.Figure:
        """Plot comprehensive performance metrics."""
        if not self.predictions:
            print("No predictions available for plotting.")
            return None

        y_true = np.array(self.predictions["y_true"])
        y_prob = np.array(self.predictions["y_prob"])

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "Final Model Performance Analysis", fontsize=16, fontweight="bold"
        )

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auroc = roc_auc_score(y_true, y_prob)
        axes[0, 0].plot(
            fpr, tpr, linewidth=2, label=f"ROC Curve (AUROC = {auroc:.3f})"
        )
        axes[0, 0].plot(
            [0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier"
        )
        axes[0, 0].set_xlabel("False Positive Rate")
        axes[0, 0].set_ylabel("True Positive Rate")
        axes[0, 0].set_title("ROC Curve")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        axes[0, 1].plot(recall, precision, linewidth=2, label="PR Curve")
        axes[0, 1].axhline(
            y=np.mean(y_true),
            color="k",
            linestyle="--",
            alpha=0.5,
            label=f"Random Classifier (AP = {np.mean(y_true):.3f})",
        )
        axes[0, 1].set_xlabel("Recall")
        axes[0, 1].set_ylabel("Precision")
        axes[0, 1].set_title("Precision-Recall Curve")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Predicted probability distribution
        axes[1, 0].hist(
            y_prob[y_true == 0],
            bins=30,
            alpha=0.7,
            label="No Recurrence",
            density=True,
        )
        axes[1, 0].hist(
            y_prob[y_true == 1],
            bins=30,
            alpha=0.7,
            label="Recurrence",
            density=True,
        )
        axes[1, 0].set_xlabel("Predicted Probability")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].set_title("Predicted Probability Distribution")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=10
        )
        axes[1, 1].plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            linewidth=2,
            label="Model",
        )
        axes[1, 1].plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
        axes[1, 1].set_xlabel("Mean Predicted Probability")
        axes[1, 1].set_ylabel("Fraction of Positives")
        axes[1, 1].set_title("Calibration Plot")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plot:
            plot_path = self.results_dir / "performance_metrics.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Performance metrics plot saved to: {plot_path}")

        return fig

    def plot_confusion_matrix(self, save_plot: bool = False) -> plt.Figure:
        """Plot confusion matrix with multiple thresholds."""
        if not self.predictions:
            print("No predictions available for confusion matrix.")
            return None

        y_true = np.array(self.predictions["y_true"])
        y_prob = np.array(self.predictions["y_prob"])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            "Confusion Matrix Analysis", fontsize=16, fontweight="bold"
        )

        thresholds = [0.5, None, 0.3]  # 0.5, optimal, 0.3
        threshold_names = [
            "Threshold = 0.5",
            "Optimal Threshold",
            "Threshold = 0.3",
        ]

        for i, (threshold, name) in enumerate(zip(thresholds, threshold_names)):
            if threshold is None:
                # Find optimal threshold (maximize F1)
                precision, recall, thresh_values = precision_recall_curve(
                    y_true, y_prob
                )
                f1_scores = (
                    2 * (precision * recall) / (precision + recall + 1e-8)
                )
                optimal_idx = np.argmax(f1_scores)
                threshold = (
                    thresh_values[optimal_idx]
                    if optimal_idx < len(thresh_values)
                    else 0.5
                )
                name = f"Optimal Threshold ({threshold:.3f})"

            y_pred = (y_prob >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)

            # Plot confusion matrix
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axes[i],
                xticklabels=["No Recurrence", "Recurrence"],
                yticklabels=["No Recurrence", "Recurrence"],
            )
            axes[i].set_title(name)
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")

            # Add metrics text
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            metrics_text = f"Sensitivity: {sensitivity:.3f}\nSpecificity: {specificity:.3f}\nAccuracy: {accuracy:.3f}"
            axes[i].text(
                0.02,
                0.98,
                metrics_text,
                transform=axes[i].transAxes,
                verticalalignment="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()

        if save_plot:
            plot_path = self.results_dir / "confusion_matrices.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrix plot saved to: {plot_path}")

        return fig

    def create_results_table(self, save_csv: bool = False) -> pd.DataFrame:
        """Create a comprehensive results table."""
        if not self.results:
            print("No results available for table creation.")
            return None

        # Extract all metrics
        data = []

        # Model info
        model_info = self.results["model_info"]
        data.append(["Model Type", f"{model_info['model_size']} ECGNet"])
        data.append(["Input Channels", model_info["input_channels"]])
        data.append(["Parameters", f"{model_info['parameters']:,}"])

        # Data splits
        splits = self.results["data_splits"]
        data.append(["Training Patients", splits["train_patients"]])
        data.append(["Validation Patients", splits["val_patients"]])
        data.append(["Test Patients", splits["test_patients"]])

        # Training info
        training = self.results["training"]
        data.append(["Total Epochs", training["total_epochs"]])
        data.append(["Best Epoch", training["best_epoch"]])
        data.append(
            ["Best Validation AUROC", f"{training['best_val_auroc']:.4f}"]
        )

        # Test performance
        perf = self.results["test_performance"]
        for metric, value in perf.items():
            if isinstance(value, float):
                data.append([metric.upper(), f"{value:.4f}"])

        df = pd.DataFrame(data, columns=["Metric", "Value"])

        if save_csv:
            csv_path = self.results_dir / "results_summary.csv"
            df.to_csv(csv_path, index=False)
            print(f"Results table saved to: {csv_path}")

        return df

    def generate_report(self, save_plots: bool = True, show_plots: bool = True):
        """Generate comprehensive analysis report."""
        print("Generating comprehensive final model analysis report...")

        # Print summary
        self.print_summary()

        # Create plots
        if self.training_history:
            fig1 = self.plot_training_history(save_plot=save_plots)
            if show_plots:
                plt.show()

        if self.predictions:
            fig2 = self.plot_performance_metrics(save_plot=save_plots)
            if show_plots:
                plt.show()

            fig3 = self.plot_confusion_matrix(save_plot=save_plots)
            if show_plots:
                plt.show()

        # Create results table
        results_table = self.create_results_table(save_csv=save_plots)
        if results_table is not None:
            print("\nDetailed Results Table:")
            print(results_table.to_string(index=False))

        print(f"\nAnalysis completed! Results saved in: {self.results_dir}")


def main():
    """Main function for final model analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze Final ECG AF Recurrence Prediction Model Results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="final_model_results",
        help="Directory containing final model results",
    )

    parser.add_argument(
        "--save-plots", action="store_true", help="Save plots to files"
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots interactively",
    )

    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print summary only, no plots",
    )

    args = parser.parse_args()

    try:
        # Create analyzer
        analyzer = FinalModelAnalyzer(args.results_dir)

        if args.summary_only:
            # Just print summary
            analyzer.print_summary()
        else:
            # Generate full report
            analyzer.generate_report(
                save_plots=args.save_plots, show_plots=not args.no_show
            )

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

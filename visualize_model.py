#!/usr/bin/env python3
"""
Create a simple architectural diagram of the ECG CNN model.

This script generates a clean, conceptual visualization of the model architecture
similar to academic paper figures, showing the high-level flow and key parameters.
"""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Add the src directory to the path so we can import our model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tang2022.model import count_parameters, create_ecg_model


def create_architecture_diagram(
    model_size="base", save_path="outputs/ecg_model_architecture.png"
):
    """
    Create a simple architectural diagram of the ECG model.

    Args:
        model_size: 'base' or 'large' model variant
        save_path: Path to save the diagram
    """

    # Create model to get actual parameters
    model = create_ecg_model(model_size, in_channels=8)
    base_channels = model.base_channels
    n_blocks = model.n_blocks

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22)
    ax.axis("off")

    # Define colors
    colors = {
        "input": "#E8E8E8",
        "bottleneck": "#D8BFD8",  # Light purple like in your image
        "dropout": "#F5DEB3",  # Wheat color
        "pooling": "#98FB98",  # Light green
        "conv": "#D8BFD8",  # Same purple
        "fc": "#FFB6C1",  # Light pink
        "output": "#E8E8E8",
    }

    # Track y position
    y_pos = 21
    box_height = 1.2
    box_width = 6
    x_center = 5
    spacing = 0.8

    def add_box(text, color, y, height=box_height, width=box_width):
        """Add a rounded rectangle box with text."""
        box = FancyBboxPatch(
            (x_center - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(
            x_center,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
        )
        return y - height / 2 - spacing

    def add_arrow(y_start, y_end):
        """Add a downward arrow between boxes."""
        ax.arrow(
            x_center,
            y_start,
            0,
            y_end - y_start + 0.1,
            head_width=0.2,
            head_length=0.1,
            fc="black",
            ec="black",
        )

    # Input
    y_pos = add_box("ECG Input: 1024 x 8 x 1", colors["input"], y_pos)

    # Input projection
    arrow_start = y_pos + spacing + box_height / 2
    y_pos = add_box("Input Projection\n(Conv1d: 8→1)", colors["conv"], y_pos)
    add_arrow(arrow_start, y_pos + box_height / 2)

    # Bottleneck blocks with correct channel and kernel progression
    block_configs = [
        (1, 16, 7),  # Block 1
        (16, 16, 7),  # Block 2
        (16, 32, 5),  # Block 3
        (32, 32, 5),  # Block 4
        (32, 64, 3),  # Block 5
        (64, 64, 3),  # Block 6
    ]

    for i, (in_ch, out_ch, kernel_size) in enumerate(block_configs):
        arrow_start = y_pos + spacing + box_height / 2
        y_pos = add_box(
            f"Bottleneck Block {i + 1}\n(kernel_size={kernel_size},\nch1={in_ch}, ch2={out_ch})",
            colors["bottleneck"],
            y_pos,
            height=1.8,
        )
        add_arrow(arrow_start, y_pos + 0.9)

        # Add dropout after every 2 blocks (like in paper image)
        if (i + 1) % 2 == 0 and i < n_blocks - 1:
            arrow_start = y_pos + 0.9 + spacing
            y_pos = add_box(
                "Dropout", colors["dropout"], y_pos - 0.4, height=0.8
            )
            add_arrow(arrow_start, y_pos + 0.4)

    # Global Average Pooling
    arrow_start = y_pos + 0.9 + spacing
    y_pos = add_box("Global Avg Pooling", colors["pooling"], y_pos - 0.4)
    add_arrow(arrow_start, y_pos + box_height / 2)

    # 1D Convolution over channel dimension (as per Tang et al. 2022)
    arrow_start = y_pos + spacing + box_height / 2
    y_pos = add_box(
        "1D Conv\n(kernel_size=1,\nin_channel=64,\nout_channel=128)\n\nBatch Norm\n\nReLU",
        colors["conv"],
        y_pos,
        height=2.8,
    )
    add_arrow(arrow_start, y_pos + 1.4)

    # Dropout
    arrow_start = y_pos + 1.4 + spacing
    y_pos = add_box("Dropout", colors["dropout"], y_pos - 0.7, height=0.8)
    add_arrow(arrow_start, y_pos + 0.4)

    # Final classifier
    arrow_start = y_pos + 0.4 + spacing
    y_pos = add_box(
        "FC\nSigmoid\n(128→1)",
        colors["fc"],
        y_pos - 0.3,
        height=1.4,
    )
    add_arrow(arrow_start, y_pos + 0.7)

    # Output
    arrow_start = y_pos + 0.7 + spacing
    y_pos = add_box("Predicted Prob.", colors["output"], y_pos - 0.3)
    add_arrow(arrow_start, y_pos + box_height / 2)

    # Add title
    ax.text(
        x_center,
        21.8,
        f"ECG CNN Architecture ({model_size.title()})",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
    )

    # Add parameter count at the bottom
    param_count = count_parameters(model)
    ax.text(
        x_center,
        0.8,
        f"Total Parameters: {param_count:,}",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
    )

    # Add notes about time dimension reduction
    ax.text(
        0.5,
        17,
        f"Time dimension reduces\nby factor of 2^{n_blocks} = {2**n_blocks}\nthrough bottleneck blocks",
        ha="left",
        va="center",
        fontsize=9,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
    )

    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Architecture diagram saved as: {save_path}")

    plt.close()


def main():
    """Main function to create model architecture diagrams."""
    print("ECG Model Architecture Visualization")
    print("=" * 40)

    # Create diagrams for different model sizes
    model_sizes = ["base", "large"]

    for model_size in model_sizes:
        print(f"\nCreating {model_size} model architecture diagram...")
        try:
            save_path = f"outputs/ecg_model_architecture_{model_size}.png"
            create_architecture_diagram(model_size, save_path)
            print("✓ Success!")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()

    print("\nArchitecture diagrams saved in outputs/ directory")


if __name__ == "__main__":
    main()

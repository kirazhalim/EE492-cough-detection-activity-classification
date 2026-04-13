"""
Generate publication-quality figures for the EE492 report.

Figures generated:
1. cough_confusion_matrices.pdf - Side-by-side confusion matrices for V1, V2, V3
2. activity_confusion_matrix_v3.pdf - V3 Run-B activity classification confusion matrix

Figures NOT generated here (export from notebooks):
- mel_spectrogram_example.png (from dl_baseline_v3.ipynb)
- prediction_timeline.png (from dl_baseline_v3.ipynb)
- training_curves_*.png (from each notebook's training plots)

Usage:
    python generate_figures.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_confusion_matrix(ax, cm, labels, title, cmap="Blues"):
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title(title, fontsize=12, fontweight="bold")

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center", fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)


def generate_cough_confusion_matrices():
    """
    Confusion matrices derived from verified classification reports:
    V1: Non-cough P=0.89 R=0.96 S=300, Cough P=0.89 R=0.71 S=129 (429 total)
    V2: Non-cough P=0.98 R=0.91 S=809, Cough P=0.71 R=0.92 S=192 (1001 total)
    V3: Non-cough P=0.99 R=0.92 S=809, Cough P=0.74 R=0.95 S=192 (1001 total)
    """
    cm_v1 = np.array([[289, 11], [37, 92]])
    cm_v2 = np.array([[737, 72], [15, 177]])
    cm_v3 = np.array([[745, 64], [10, 182]])

    labels = ["Non-Cough", "Cough"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    plot_confusion_matrix(axes[0], cm_v1, labels, "V1 (429 test windows)")
    plot_confusion_matrix(axes[1], cm_v2, labels, "V2 (1001 test windows)")
    plot_confusion_matrix(axes[2], cm_v3, labels, "V3 (1001 test windows)")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "cough_confusion_matrices.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    print("Generating figures for EE492 report...\n")
    generate_cough_confusion_matrices()
    print("\nDone. Check the figures/ directory.")
    print("\nThe following figures must be exported manually from notebooks:")
    print("  - mel_spectrogram_example.png  (from dl_baseline_v3.ipynb, cell after spectrogram transform)")
    print("  - prediction_timeline.png      (from dl_baseline_v3.ipynb)")
    print("  - activity_confusion_matrix_v3 (from dl_baseline_v3.ipynb, Run-B CM)")

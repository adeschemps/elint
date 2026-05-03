import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

# PDW dim layout for the alan-turing-institute synthetic radar dataset.
# https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset
TOA_DIM, FREQ_DIM, PW_DIM, DOA_DIM, AMP_DIM = 0, 1, 2, 3, 4


def plot_sequence(seq: torch.Tensor | np.ndarray) -> Figure:
    """Plot a single PDW sequence on a 2x2 grid.

    Panels: TOA vs (frequency, PW, DOA, PRI). PRI is derived as the diff of TOA.

    Args:
        seq: tensor or array of shape (seq_len, 5).
    """
    arr = seq.numpy() if isinstance(seq, torch.Tensor) else seq

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    panels = [
        (axes[0, 0], "frequency (MHz)", FREQ_DIM),
        (axes[0, 1], "PW (us)", PW_DIM),
        (axes[1, 0], "DOA (deg)", DOA_DIM),
    ]
    for ax, ylabel, idx in panels:
        ax.scatter(arr[:, TOA_DIM], arr[:, idx], alpha=0.5, s=10)
        ax.set_xlabel("TOA (us)")
        ax.set_ylabel(ylabel)

    ax = axes[1, 1]
    ax.scatter(arr[1:, TOA_DIM], np.diff(arr[:, TOA_DIM]), alpha=0.5, s=10)
    ax.set_xlabel("TOA (us)")
    ax.set_ylabel("PRI (us)")

    fig.tight_layout()
    return fig

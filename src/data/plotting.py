import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

# PDW dim layout for the alan-turing-institute synthetic radar dataset.
# https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset
TOA_DIM, FREQ_DIM, PW_DIM, DOA_DIM, AMP_DIM = 0, 1, 2, 3, 4


def _denormalize(
    seq: torch.Tensor | np.ndarray, stats: dict[str, float]
) -> np.ndarray:
    arr = (seq.detach().cpu().numpy() if isinstance(seq, torch.Tensor) else seq).astype(
        np.float64, copy=True
    )
    arr[:, TOA_DIM] = (
        arr[:, TOA_DIM] * (stats["toa_max"] - stats["toa_min"]) + stats["toa_min"]
    )
    arr[:, FREQ_DIM] = arr[:, FREQ_DIM] * stats["freq_std"] + stats["freq_mean"]
    arr[:, PW_DIM] = arr[:, PW_DIM] * stats["pw_std"] + stats["pw_mean"]
    arr[:, DOA_DIM] = arr[:, DOA_DIM] * 360.0
    arr[:, AMP_DIM] = arr[:, AMP_DIM] * stats["amp_std"] + stats["amp_mean"]
    return arr


def plot_sequence(
    seq: torch.Tensor | np.ndarray, stats: dict[str, float]
) -> Figure:
    """Plot a single PDW sequence on a 2x2 grid.

    Panels: TOA vs (frequency, PW, DOA, PRI). PRI is derived as the diff of TOA.

    Args:
        seq: tensor or array of shape (seq_len, 5), in normalized units.
        stats: per-sequence stats from the dataset, used to recover original units.
    """
    arr = _denormalize(seq, stats)

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


def plot_sequence_prediction(
    true: torch.Tensor | np.ndarray,
    pred: torch.Tensor | np.ndarray,
    stats: dict[str, float],
) -> Figure:
    """Plot true vs predicted PDW sequence on a 2x2 grid.

    True values in blue, predicted values in red. Both inputs are denormalized
    using *stats* before plotting.

    Args:
        true: tensor or array of shape (seq_len, 5), in normalized units.
        pred: tensor or array of shape (seq_len, 5), in normalized units.
        stats: per-sequence stats from the dataset.
    """
    true_arr = _denormalize(true, stats)
    pred_arr = _denormalize(pred, stats)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    panels = [
        (axes[0, 0], "frequency (MHz)", FREQ_DIM),
        (axes[0, 1], "PW (us)", PW_DIM),
        (axes[1, 0], "DOA (deg)", DOA_DIM),
    ]
    for ax, ylabel, idx in panels:
        ax.scatter(true_arr[:, TOA_DIM], true_arr[:, idx], color="blue", alpha=0.5, s=10, label="true")
        ax.scatter(pred_arr[:, TOA_DIM], pred_arr[:, idx], color="red", alpha=0.5, s=10, label="predicted")
        ax.set_xlabel("TOA (us)")
        ax.set_ylabel(ylabel)
        ax.legend()

    ax = axes[1, 1]
    ax.scatter(
        true_arr[1:, TOA_DIM],
        np.diff(true_arr[:, TOA_DIM]),
        color="blue",
        alpha=0.5,
        s=10,
        label="true",
    )
    ax.scatter(
        pred_arr[1:, TOA_DIM],
        np.diff(pred_arr[:, TOA_DIM]),
        color="red",
        alpha=0.5,
        s=10,
        label="predicted",
    )
    ax.set_xlabel("TOA (us)")
    ax.set_ylabel("PRI (us)")
    ax.legend()

    fig.tight_layout()
    return fig

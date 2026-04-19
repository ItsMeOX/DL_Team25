"""Evaluation and visualization helpers for anomaly detection notebooks."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from torch.utils.data import DataLoader, Subset


def infer_case_from_path(path: str) -> str:
    """Extract case id from data path."""
    parts = path.replace("\\", "/").split("/")
    if "IND" in parts:
        ind_pos = parts.index("IND")
        if ind_pos + 1 < len(parts):
            return parts[ind_pos + 1]
    return "unknown"


def count_ind_cnt(subset: Subset, sample_table: Sequence[Tuple[str, int, int]]) -> Tuple[int, int]:
    """Count IND and CNT samples inside a subset using `start` metadata."""
    ind_count = 0
    cnt_count = 0
    for idx in subset.indices:
        _, start, _ = sample_table[idx]
        if start is None:
            ind_count += 1
        else:
            cnt_count += 1
    return ind_count, cnt_count


def build_case_scoped_loaders(
    dataset,
    sample_cases: Sequence[str],
    normal_indices: Sequence[int],
    anom_indices: Sequence[int],
    selected_cases: Iterable[str],
    batch_size: int = 8,
    anom_ratio: float = 1.0,
):
    """Create normal/anomaly dataloaders filtered by selected cases."""
    selected = set(selected_cases)

    sel_normal_idx = [i for i in normal_indices if sample_cases[i] in selected]
    sel_anom_idx = [i for i in anom_indices if sample_cases[i] in selected]

    n_normal = len(sel_normal_idx)
    target_anom = max(1, int(n_normal * anom_ratio)) if n_normal else 0
    if target_anom > 0 and len(sel_anom_idx) > target_anom:
        sel_anom_idx = random.sample(sel_anom_idx, target_anom)

    normal_subset = Subset(dataset, sel_normal_idx)
    anom_subset = Subset(dataset, sel_anom_idx)

    normal_loader = DataLoader(normal_subset, batch_size=batch_size, shuffle=False)
    anom_loader = DataLoader(anom_subset, batch_size=batch_size, shuffle=False)
    return normal_loader, anom_loader, len(sel_normal_idx), len(sel_anom_idx)


def build_scope_loader_dict(
    dataset,
    sample_cases: Sequence[str],
    normal_indices: Sequence[int],
    anom_indices: Sequence[int],
    case_scopes: Dict[str, Sequence[str]],
    batch_size: int = 8,
    anom_ratio: float = 1.0,
    prefix: str = "val",
):
    """Build loaders for each named case scope."""
    scoped = {}
    for scope_name, scope_cases in case_scopes.items():
        n_loader, a_loader, n_normal, n_anom = build_case_scoped_loaders(
            dataset=dataset,
            sample_cases=sample_cases,
            normal_indices=normal_indices,
            anom_indices=anom_indices,
            selected_cases=scope_cases,
            batch_size=batch_size,
            anom_ratio=anom_ratio,
        )
        scoped[scope_name] = {
            f"{prefix}_normal_loader": n_loader,
            f"{prefix}_anom_loader": a_loader,
            "n_normal": n_normal,
            "n_anom": n_anom,
        }
    return scoped


def get_reconstruction_scores(model, loader, device, score_type: str = "l1") -> np.ndarray:
    """Compute reconstruction error score for each sample in loader."""
    model.eval()
    scores: List[float] = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            recon = model(x)
            if score_type == "l1":
                err = torch.mean(torch.abs(x - recon), dim=(1, 2, 3))
            else:
                err = torch.mean((x - recon) ** 2, dim=(1, 2, 3))
            scores.extend(err.detach().cpu().numpy())
    return np.array(scores)


def find_best_f1_threshold(normal_scores: np.ndarray, anom_scores: np.ndarray) -> float:
    """Find threshold that maximizes F1 on a validation split."""
    y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anom_scores))])
    y_score = np.concatenate([normal_scores, anom_scores])
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return float(np.mean(y_score))
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = int(np.argmax(f1[:-1]))
    return float(thresholds[best_idx])


def evaluate_scores(normal_scores: np.ndarray, anom_scores: np.ndarray, threshold: float):
    """Return PR-AUC, F1, and mean-gap from score arrays."""
    y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anom_scores))])
    y_score = np.concatenate([normal_scores, anom_scores])
    pr_auc = average_precision_score(y_true, y_score)
    f1 = f1_score(y_true, (y_score > threshold).astype(int))
    gap = float(np.mean(anom_scores) - np.mean(normal_scores))
    return pr_auc, f1, gap


def downsample_anomaly_scores(
    normal_scores: np.ndarray,
    anom_scores: np.ndarray,
    anom_ratio: float = 0.05,
) -> np.ndarray:
    """Downsample anomaly scores to a target ratio against normal."""
    n_normal = len(normal_scores)
    target_anom = max(1, int(n_normal * anom_ratio))
    if len(anom_scores) > target_anom:
        sampled_idx = random.sample(range(len(anom_scores)), target_anom)
        return anom_scores[sampled_idx]
    return anom_scores


def plot_multichannel_spec(x, title: str = "Spectrogram", is_error: bool = False) -> None:
    """Plot each channel of a spectrogram tensor in stacked subplots."""
    x_np = x.cpu().numpy()
    channels = x_np.shape[0]
    fig, axs = plt.subplots(channels, 1, figsize=(10, 2 * channels), sharex=True)
    if channels == 1:
        axs = [axs]

    im = None
    for i in range(channels):
        if is_error:
            vmin, vmax = 0, x_np[i].max()
        else:
            vmin, vmax = 0, 1
        im = axs[i].imshow(
            x_np[i],
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        axs[i].set_ylabel(f"Ch {i + 1}")
        axs[i].set_yticks([])

    axs[-1].set_xlabel("Time")
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1.05, 1])
    fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02)
    plt.show()


def plot_one_reconstruction(model, loader_iter, device) -> None:
    """Plot original and reconstructed spectrogram for one sample."""
    model.eval()
    x, _ = next(loader_iter)
    x = x.to(device)

    with torch.no_grad():
        recon = model(x)

    x0 = x[0].cpu()
    recon0 = recon[0].cpu()

    error_map = torch.abs(x0 - recon0)

    err = error_map.mean().item()

    plot_multichannel_spec(x0, "Original Log-Mel Spectrogram (Sample)")
    plot_multichannel_spec(recon0, f"Reconstruction | L1 Error: {err:.4f}")
    plot_multichannel_spec(
        error_map,
        f"Error Map | Mean Error: {err:.4f}",
        is_error=True
    )
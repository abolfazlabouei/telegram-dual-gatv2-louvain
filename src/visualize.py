"""
All plotting in one place. Every function takes an optional `save_path`;
if given, the figure is saved there and closed (headless-friendly, e.g.
running on a remote GPU box over SSH); if omitted, it's shown interactively
as in the original notebook.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def _finish(save_path: str | None) -> None:
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()


def plot_degree_distribution(degrees: np.ndarray, save_path: str | None = None) -> None:
    plt.figure()
    plt.hist(degrees, bins=100)
    plt.xscale("log"); plt.yscale("log")
    plt.title("Degree Distribution (log-log)")
    plt.xlabel("Degree"); plt.ylabel("Count")
    _finish(save_path)


def plot_weight_distribution(weights: np.ndarray, save_path: str | None = None) -> None:
    plt.figure()
    plt.hist(weights, bins=100)
    plt.title("Edge Weight Distribution (raw)")
    plt.xlabel("Weight"); plt.ylabel("Count")
    _finish(save_path)

    if save_path:
        base, ext = os.path.splitext(save_path)
        save_path_log = f"{base}_log1p{ext}"
    else:
        save_path_log = None
    plt.figure()
    plt.hist(np.log1p(weights), bins=100)
    plt.title("Edge Weight Distribution (log1p)")
    plt.xlabel("log1p(Weight)"); plt.ylabel("Count")
    _finish(save_path_log)


def plot_component_sizes(sizes: np.ndarray, save_path: str | None = None) -> None:
    plt.figure()
    plt.hist(sizes, bins=100)
    plt.yscale("log")
    plt.title("Connected Component Size Distribution")
    plt.xlabel("Component size"); plt.ylabel("Count")
    _finish(save_path)


def plot_degree_ccdf(xs: np.ndarray, ccdf: np.ndarray, save_path: str | None = None) -> None:
    plt.figure(figsize=(5, 4))
    plt.plot(xs, ccdf, marker=".", linestyle="none")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Degree (k)"); plt.ylabel("CCDF P(K>=k)")
    plt.title("Degree CCDF (log-log)")
    _finish(save_path)


def plot_rank_size(degrees: np.ndarray, save_path: str | None = None) -> None:
    d_sorted = np.sort(degrees[degrees > 0])[::-1]
    ranks = np.arange(1, len(d_sorted) + 1)
    plt.figure(figsize=(5, 4))
    plt.plot(ranks, d_sorted, ".", alpha=0.6)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Rank"); plt.ylabel("Degree")
    plt.title("Rank-Size (Zipf)")
    _finish(save_path)


def plot_clustering_vs_degree(xs: np.ndarray, ys: np.ndarray, save_path: str | None = None) -> None:
    if len(xs) == 0:
        return
    plt.figure(figsize=(5, 4))
    plt.plot(xs, ys, "-o")
    plt.xscale("log")
    plt.xlabel("Degree (k)"); plt.ylabel("Avg clustering")
    plt.title("Clustering vs Degree")
    _finish(save_path)


def plot_k_core_curve(ks: list[int], sizes: list[int], save_path: str | None = None) -> None:
    plt.figure(figsize=(5, 4))
    plt.plot(ks, sizes, "-o")
    plt.xlabel("k"); plt.ylabel("#nodes in k-core")
    plt.yscale("log")
    plt.title("k-core size curve")
    _finish(save_path)


def plot_power_law_fit(fit_result: dict, save_path: str | None = None) -> None:
    """`fit_result` is the dict returned by `graph_stats.power_law_fit`."""
    fit = fit_result["fit_object"]
    gamma = fit_result["alpha"]
    plt.figure(figsize=(5, 4))
    fit.plot_ccdf(color="k", linewidth=2, label="Empirical")
    fit.power_law.plot_ccdf(color="r", linestyle="--", label=f"Power-law (alpha={gamma:.2f})")
    fit.lognormal.plot_ccdf(color="b", linestyle=":", label="Lognormal")
    plt.xscale("log"); plt.yscale("log"); plt.legend()
    plt.xlabel("Degree (k)"); plt.ylabel("CCDF")
    plt.title("Degree tail fit (CCDF)")
    _finish(save_path)


def plot_alpha_histogram(alpha: np.ndarray, save_path: str | None = None) -> None:
    plt.figure()
    plt.hist(alpha, bins=50)
    plt.title("Fusion Gate alpha Distribution")
    plt.xlabel("alpha (weight on structural stream)"); plt.ylabel("Count")
    _finish(save_path)


def plot_community_sizes(pred: np.ndarray, title: str = "Community Size Distribution",
                          save_path: str | None = None) -> None:
    sizes = np.bincount(pred)
    plt.figure()
    plt.hist(sizes[sizes > 0], bins=100)
    plt.yscale("log")
    plt.title(title)
    plt.xlabel("Community size"); plt.ylabel("Count (log)")
    _finish(save_path)


def plot_k_sensitivity(df: pd.DataFrame, save_path: str | None = None) -> None:
    """`df` as returned by `clustering.k_sensitivity_sweep` (columns k, NMI, ARI, Purity)."""
    for metric in ["NMI", "ARI", "Purity"]:
        plt.figure()
        plt.plot(df["k"], df[metric], marker="o")
        plt.title(f"{metric} vs k")
        plt.xlabel("k"); plt.ylabel(metric)
        this_path = None
        if save_path:
            base, ext = os.path.splitext(save_path)
            this_path = f"{base}_{metric.lower()}{ext}"
        _finish(this_path)


def plot_tsne(X: np.ndarray, labels: np.ndarray | None = None, n_samples: int = 3000,
              seed: int = 42, save_path: str | None = None) -> None:
    """Optional exploratory plot -- 2D t-SNE of a random subsample of the
    text-embedding matrix, colored by reference label if provided."""
    from sklearn.manifold import TSNE

    n_samples = min(n_samples, len(X))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), n_samples, replace=False)
    X_sample = X[idx]

    tsne = TSNE(n_components=2, perplexity=30, random_state=seed)
    X_tsne = tsne.fit_transform(X_sample)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        from sklearn.preprocessing import LabelEncoder
        labels_num = LabelEncoder().fit_transform(labels[idx])
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_num, cmap="tab20", s=10, alpha=0.8)
        plt.colorbar(scatter, label="Label ID")
        plt.title("t-SNE of text embeddings (colored by label)")
    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=5, alpha=0.7)
        plt.title("t-SNE of text embeddings (sample)")
    _finish(save_path)

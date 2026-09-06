"""
Descriptive statistics for a built graph: connected components, degree
distribution, density, clustering, assortativity, k-core structure, and an
optional power-law fit on the degree tail.

Plotting is kept out of this module on purpose (see `src/visualize.py`) so
these functions can also be used non-interactively / headlessly.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def connected_components_report(edges_idx: np.ndarray, num_nodes: int) -> dict:
    """`edges_idx` is an (E, 2) array of 0-based node indices (src, dst)."""
    row = edges_idx[:, 0]
    col = edges_idx[:, 1]
    data = np.ones_like(row, dtype=np.int8)
    A = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A = A + A.T
    A.data = np.ones_like(A.data)

    n_comp, labels_cc = connected_components(A.tocsr(), directed=False, return_labels=True)
    sizes = np.bincount(labels_cc)
    sizes_sorted = np.sort(sizes)[::-1]

    return {
        "num_components": int(n_comp),
        "component_sizes": sizes,
        "top10_component_sizes": sizes_sorted[:10].tolist(),
        "giant_component_fraction": float(sizes_sorted[0] / num_nodes) if num_nodes else 0.0,
    }


def basic_network_stats(
    G: nx.Graph,
    clustering_sample_size: int = 2000,
    clustering_max_degree: int = 2000,
) -> dict:
    """Node/edge counts, density, degree stats, clustering, assortativity.

    Average clustering coefficient is exact only for small graphs. Computing
    it exactly means checking every pair of each node's neighbors -- for a
    node with degree k that's O(k^2), and `nx.average_clustering` does this
    for *every* node. Real membership graphs are heavy-tailed (a handful of
    "hub" groups with degree in the thousands or tens of thousands), and a
    single such node can already cost tens of millions of pair-checks; doing
    this for all of them can run for hours on a large graph.

    Instead: exclude nodes above `clustering_max_degree` (hubs -- typically
    near-zero clustering anyway, and the main cost driver) and estimate the
    average from a random sample of up to `clustering_sample_size` of the
    remaining nodes. This is the standard "sampled average clustering"
    approach for large graphs and is noted as an estimate in the returned
    dict so it's transparent if you paste it into a table.
    """
    import random

    N = G.number_of_nodes()
    M = G.number_of_edges()
    degs = np.array([d for _, d in G.degree()])
    deg_mean = float(degs.mean()) if N else 0.0
    deg_std = float(degs.std()) if N else 0.0
    density = (2 * M / (N * (N - 1))) if N > 1 else 0.0

    components = list(nx.connected_components(G))
    num_comp = len(components)
    gcc_size = max((len(c) for c in components), default=0)

    if N:
        candidates = [n for n, d in G.degree() if d <= clustering_max_degree]
        excluded = N - len(candidates)
        if len(candidates) > clustering_sample_size:
            sample_nodes = random.sample(candidates, clustering_sample_size)
        else:
            sample_nodes = candidates
        if sample_nodes:
            clust_avg = nx.average_clustering(G, nodes=sample_nodes, count_zeros=True)
            clust_note = (f"estimated from {len(sample_nodes):,} sampled nodes "
                          f"(degree <= {clustering_max_degree:,}; {excluded:,} hub node(s) excluded)")
        else:
            clust_avg = float("nan")
            clust_note = "skipped: every node exceeds the degree cap"
    else:
        clust_avg = 0.0
        clust_note = "n/a (empty graph)"

    assort = nx.degree_pearson_correlation_coefficient(G) if M else float("nan")

    return {
        "num_nodes": N,
        "num_edges": M,
        "avg_degree": (2 * M / N) if N else 0.0,
        "degree_mean": deg_mean,
        "degree_std": deg_std,
        "degree_min": int(degs.min()) if N else 0,
        "degree_max": int(degs.max()) if N else 0,
        "density": density,
        "num_components": num_comp,
        "giant_component_size": gcc_size,
        "avg_clustering": clust_avg,
        "avg_clustering_note": clust_note,
        "assortativity": assort,
    }


def degree_ccdf(degrees: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Complementary CDF of the degree sequence, for log-log plots."""
    deg_pos = degrees[degrees > 0]
    if len(deg_pos) == 0:
        return np.array([]), np.array([])
    cnt = Counter(deg_pos)
    xs = np.array(sorted(cnt.keys()))
    cdf = np.cumsum([cnt[x] for x in xs]) / len(deg_pos)
    ccdf = 1 - cdf
    return xs, ccdf


def clustering_vs_degree(
    G: nx.Graph, n_bins: int = 15, sample_size: int = 5000, max_degree: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Average local clustering coefficient binned by (log-spaced) degree.

    Same cost problem as `basic_network_stats`'s clustering estimate (a
    degree-k node costs O(k^2) to compute exactly) -- bounded here the same
    way: nodes above `max_degree` are excluded, and the rest are capped to
    a random sample of `sample_size` before computing per-node clustering.
    """
    import random

    all_nodes = list(G.nodes())
    candidates = [n for n in all_nodes if G.degree(n) <= max_degree]
    if len(candidates) > sample_size:
        candidates = random.sample(candidates, sample_size)

    k_vals = np.array([G.degree(n) for n in candidates])
    c_vals = np.array(list(nx.clustering(G, nodes=candidates).values()))
    if len(k_vals) == 0:
        return np.array([]), np.array([])
    deg_min, deg_max = max(1, int(k_vals.min())), max(2, int(k_vals.max()))
    bins = np.unique(np.round(np.logspace(np.log10(deg_min), np.log10(deg_max), n_bins))).astype(int)
    bins = np.unique(np.clip(bins, 1, None))
    xs, ys = [], []
    for i in range(len(bins) - 1):
        mask = (k_vals >= bins[i]) & (k_vals < bins[i + 1])
        if mask.sum() > 50:
            xs.append((bins[i] + bins[i + 1]) / 2)
            ys.append(c_vals[mask].mean())
    return np.array(xs), np.array(ys)


def k_core_curve(G: nx.Graph, max_k: int = 100) -> tuple[list[int], list[int]]:
    """Number of nodes remaining in the k-core for increasing k.

    Uses `nx.core_number` -- one linear-ish peeling pass over the whole
    graph -- rather than calling `nx.k_core` separately for each k value.
    The original called `nx.k_core` (itself an O(V+E)-ish scan) up to
    `max_k` times *from scratch*, which for a large, hub-heavy graph (many
    edges to rescan every time) was a second real bottleneck alongside the
    clustering-coefficient one. Computing each node's core number once and
    then just counting how many are >= k for each k is dramatically
    cheaper and gives the exact same curve.
    """
    H = G.copy()
    H.remove_edges_from(nx.selfloop_edges(H))
    if H.number_of_nodes() == 0:
        return [], []

    core_numbers = np.array(list(nx.core_number(H).values()))
    deg_max = int(core_numbers.max()) if len(core_numbers) else 0

    ks, sizes = [], []
    for k in range(1, min(max_k, deg_max + 1) + 1):
        size = int((core_numbers >= k).sum())
        if size == 0:
            break
        ks.append(k)
        sizes.append(size)
    return ks, sizes


def power_law_fit(degrees: np.ndarray) -> dict | None:
    """Fit a power law to the degree tail with the `powerlaw` package and
    compare it against lognormal/exponential alternatives.

    Returns None (and prints a note) if the `powerlaw` package is not
    installed -- it is an optional, occasionally slow-to-build dependency.
    """
    try:
        import powerlaw  # noqa: F401 -- optional dependency
    except ImportError:
        print("[graph_stats] `powerlaw` package not installed; skipping power-law fit "
              "(pip install powerlaw to enable).")
        return None

    deg_pos = degrees[degrees > 0]
    if len(deg_pos) == 0:
        return None

    fit = powerlaw.Fit(deg_pos, discrete=True, verbose=False)
    gamma = fit.power_law.alpha
    kmin = fit.power_law.xmin
    R_pl_logn, p_pl_logn = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
    R_pl_exp, p_pl_exp = fit.distribution_compare("power_law", "exponential", normalized_ratio=True)

    return {
        "alpha": float(gamma),
        "xmin": float(kmin),
        "R_powerlaw_vs_lognormal": float(R_pl_logn),
        "p_powerlaw_vs_lognormal": float(p_pl_logn),
        "R_powerlaw_vs_exponential": float(R_pl_exp),
        "p_powerlaw_vs_exponential": float(p_pl_exp),
        "fit_object": fit,  # kept for plotting (fit.plot_ccdf, etc.)
    }

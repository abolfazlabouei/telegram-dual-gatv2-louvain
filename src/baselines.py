"""
Classical community-detection baselines for the Chapter 5 comparison
table: Louvain, Leiden, Infomap, and Label Propagation, each run in a
weighted and an unweighted variant on the structural graph.

Dependencies:
  - Louvain, Label Propagation: only need `networkx` / `python-louvain`,
    already in requirements.txt.
  - Leiden: `pip install python-igraph leidenalg`
  - Infomap: `pip install infomap`

Leiden and Infomap are optional -- if their package isn't installed,
`run_all_classical_baselines` skips that algorithm with a printed note
instead of crashing, so you can still get Louvain + Label Propagation
numbers immediately and add the others later.

For every algorithm, "weighted" vs "unweighted" controls whether the
*community-detection step itself* sees real edge weights or a constant
weight of 1 -- but Modularity/NMI/ARI/Purity for every resulting
partition are always computed against the *true* weighted graph, so
numbers are comparable across the weighted/unweighted split (this
matches evaluating "how good is this partition, really" independent of
what the algorithm was allowed to look at).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import networkx as nx
from community import best_partition
from networkx.algorithms.community import asyn_lpa_communities

from .evaluate import eval_partition


def _unweighted_copy(G: nx.Graph) -> nx.Graph:
    """Same nodes/edges, every weight forced to 1.0 -- used so each
    algorithm's own weight-handling quirks don't matter: they all just
    read the `weight` attribute, which is uniformly 1 here."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from((u, v, {"weight": 1.0}) for u, v in G.edges())
    return H


def _partition_from_communities(communities, num_nodes: int) -> dict:
    """networkx community functions return an iterable of node sets;
    convert to the {node: community_id} dict used everywhere else here."""
    partition: dict = {}
    for cid, nodes in enumerate(communities):
        for n in nodes:
            partition[n] = cid
    next_id = (max(partition.values()) + 1) if partition else 0
    for n in range(num_nodes):
        if n not in partition:
            partition[n] = next_id
            next_id += 1
    return partition


def run_louvain_baseline(G: nx.Graph, weighted: bool, resolution: float = 1.0,
                          seed: int = 42) -> dict:
    G_use = G if weighted else _unweighted_copy(G)
    t0 = time.time()
    partition = best_partition(G_use, weight="weight", resolution=resolution, random_state=seed)
    return {"partition": partition, "runtime_s": time.time() - t0}


def run_label_propagation_baseline(G: nx.Graph, weighted: bool, seed: int = 42, **_) -> dict:
    G_use = G if weighted else _unweighted_copy(G)
    t0 = time.time()
    communities = list(asyn_lpa_communities(G_use, weight="weight", seed=seed))
    partition = _partition_from_communities(communities, G.number_of_nodes())
    return {"partition": partition, "runtime_s": time.time() - t0}


def run_leiden_baseline(G: nx.Graph, weighted: bool, resolution: float = 1.0,
                         seed: int = 42) -> dict:
    """Needs `pip install python-igraph leidenalg`."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError as e:
        raise ImportError(
            "Leiden baseline needs `pip install python-igraph leidenalg`"
        ) from e

    G_use = G if weighted else _unweighted_copy(G)
    nodes = list(G_use.nodes())
    node2idx = {n: i for i, n in enumerate(nodes)}
    edge_list = list(G_use.edges())
    ig_graph = ig.Graph(n=len(nodes), edges=[(node2idx[u], node2idx[v]) for u, v in edge_list])
    ig_graph.es["weight"] = [G_use[u][v].get("weight", 1.0) for u, v in edge_list]

    t0 = time.time()
    partition_obj = leidenalg.find_partition(
        ig_graph, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution, seed=seed,
    )
    runtime = time.time() - t0

    partition = {}
    for cid, comm in enumerate(partition_obj):
        for idx in comm:
            partition[nodes[idx]] = cid
    return {"partition": partition, "runtime_s": runtime}


def run_infomap_baseline(G: nx.Graph, weighted: bool, seed: int = 42, **_) -> dict:
    """Needs `pip install infomap`."""
    try:
        from infomap import Infomap
    except ImportError as e:
        raise ImportError("Infomap baseline needs `pip install infomap`") from e

    G_use = G if weighted else _unweighted_copy(G)
    im = Infomap(f"--two-level --seed {seed} --silent")

    t0 = time.time()
    for u, v, d in G_use.edges(data=True):
        im.add_link(int(u), int(v), float(d.get("weight", 1.0)))
    im.run()
    runtime = time.time() - t0

    partition = {}
    for node in im.tree:
        if node.is_leaf:
            partition[node.node_id] = node.module_id
    next_id = (max(partition.values()) + 1) if partition else 0
    for n in G_use.nodes():
        if n not in partition:
            partition[n] = next_id
            next_id += 1
    return {"partition": partition, "runtime_s": runtime}


_ALGORITHMS = {
    "louvain": run_louvain_baseline,
    "label_propagation": run_label_propagation_baseline,
    "leiden": run_leiden_baseline,
    "infomap": run_infomap_baseline,
}


def run_all_classical_baselines(
    G: nx.Graph,
    y_true: np.ndarray,
    resolution: float = 1.0,
    seed: int = 42,
    algorithms: list[str] | None = None,
) -> pd.DataFrame:
    """Run every algorithm in `algorithms` (default: all four) in both
    weighted and unweighted mode, evaluating each resulting partition
    against `y_true` and the true weighted `G`. Any algorithm whose
    optional dependency isn't installed is skipped with a printed note,
    not a crash. Returns one row per (algorithm, weighted) run, in the
    same shape as the thesis's Chapter 5 comparison table."""
    algorithms = algorithms or list(_ALGORITHMS.keys())
    rows = []

    for algo_name in algorithms:
        fn = _ALGORITHMS[algo_name]
        for weighted in (True, False):
            label = f"{algo_name} ({'weighted' if weighted else 'unweighted'})"
            try:
                result = fn(G, weighted=weighted, resolution=resolution, seed=seed)
            except ImportError as e:
                print(f"[baselines] skipping {label}: {e}")
                continue

            partition = result["partition"]
            Q, NMI, ARI, PUR, K, _ = eval_partition(G, partition, y_true)
            rows.append({
                "algorithm": algo_name, "weighted": weighted,
                "Modularity": Q, "NMI": NMI, "ARI": ARI, "Purity": PUR,
                "num_communities": K, "runtime_s": result["runtime_s"],
            })
            print(f"[baselines] {label}: Q={Q:.4f} NMI={NMI:.4f} ARI={ARI:.4f} "
                  f"Purity={PUR:.4f} #C={K} time={result['runtime_s']:.1f}s")

    return pd.DataFrame(rows)

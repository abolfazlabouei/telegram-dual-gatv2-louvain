"""
Turn fused node embeddings into communities:

  1. build a (mutual) cosine k-NN graph over the fused embeddings,
  2. run Louvain on that reconstructed graph,
  3. evaluate against the reference `label` column with NMI / ARI / Purity
     / Modularity (see `src/evaluate.py`).

Also provides a structural-only Louvain baseline (Louvain run directly on
the membership graph, no learned embeddings at all) for comparison.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd
from community import best_partition, modularity as louvain_modularity

from .graphs import knn_edges
from .evaluate import eval_partition


def build_fused_knn_graph(emb: np.ndarray, k: int, mutual: bool = True) -> nx.Graph:
    """(Mutual) cosine k-NN graph over the fused embeddings, as an
    undirected weighted `networkx.Graph` ready for Louvain."""
    rows, cols, ws = knn_edges(emb, k, mutual)

    G = nx.Graph()
    G.add_nodes_from(range(len(emb)))
    for u, v, w in zip(rows, cols, ws):
        if G.has_edge(u, v):
            if G[u][v]["weight"] < w:
                G[u][v]["weight"] = w
        else:
            G.add_edge(u, v, weight=w)
    return G


def run_louvain(G: nx.Graph, resolution: float = 1.0, seed: int = 42) -> dict:
    """Returns a {node_index: community_id} partition dict."""
    return best_partition(G, weight="weight", resolution=resolution, random_state=seed)


def pyg_to_nx_weighted(data_cpu) -> nx.Graph:
    """Convert a PyG `Data(edge_index, edge_attr)` object into an undirected
    weighted networkx graph (duplicate/parallel edges keep the max weight).

    Explicitly adds every node 0..num_nodes-1 first -- otherwise a node
    with zero edges (isolated) would silently be missing from the graph,
    undercounting `G.number_of_nodes()` relative to the true dataset size
    in any stats table built from it.
    """
    G = nx.Graph()
    G.add_nodes_from(range(data_cpu.num_nodes))
    src = data_cpu.edge_index[0].cpu().numpy()
    dst = data_cpu.edge_index[1].cpu().numpy()
    w = data_cpu.edge_attr.cpu().numpy().astype(float)
    for u, v, ww in zip(src, dst, w):
        u, v = int(u), int(v)
        if G.has_edge(u, v):
            if G[u][v]["weight"] < ww:
                G[u][v]["weight"] = ww
        else:
            G.add_edge(u, v, weight=ww)
    return G


def structural_only_baseline(
    data_struct_cpu,
    y_true: np.ndarray,
    resolution: float = 1.0,
    seed: int = 42,
) -> dict:
    """Louvain run directly on the structural (membership) graph, with no
    learned embeddings involved at all -- a sanity-check baseline."""
    G_struct = pyg_to_nx_weighted(data_struct_cpu)
    partition = best_partition(G_struct, weight="weight", resolution=resolution, random_state=seed)
    Q, NMI, ARI, PUR, K, pred = eval_partition(G_struct, partition, y_true)
    return {"Q": Q, "NMI": NMI, "ARI": ARI, "Purity": PUR, "num_communities": K,
            "partition": partition, "pred": pred, "graph": G_struct}


def k_sensitivity_sweep(
    emb: np.ndarray,
    y_true: np.ndarray,
    k_list: list[int],
    mutual: bool,
    resolution: float,
    seed: int,
) -> pd.DataFrame:
    """Re-run the fused-kNN-graph -> Louvain -> evaluate pipeline for each k
    in `k_list`, returning a tidy DataFrame of NMI/ARI/Purity per k."""
    rows = []
    for k in k_list:
        G = build_fused_knn_graph(emb, k, mutual)
        partition = run_louvain(G, resolution=resolution, seed=seed)
        Q, NMI, ARI, PUR, K_comm, _ = eval_partition(G, partition, y_true)
        rows.append({"k": k, "Q": Q, "NMI": NMI, "ARI": ARI, "Purity": PUR,
                     "num_communities": K_comm})
    return pd.DataFrame(rows)

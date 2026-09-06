"""
Build the two complementary graphs the Dual-GATv2 model consumes:

  - the *structural* graph, from shared-membership edge weights
    (`telegram_graph.edgelist`, already thresholded at >=5 shared users
    upstream of this repo);
  - the *text-similarity* graph, a (mutual) k-nearest-neighbour graph over
    the cosine similarity of each group's text embedding.

Both graphs are built over the exact same 0..N-1 node indexing so they can
share node features and be trained with aligned mini-batches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data


def build_structural_graph(
    edges: pd.DataFrame,
    id2idx: dict,
    reverse_rate: float,
    reverse_rate_threshold: float = 0.8,
) -> Data:
    """Min-max-normalized (via log1p) edge weights, symmetrized if the edge
    list does not already store both directions.

    `reverse_rate` should come from `data.compute_reverse_rate`: if fewer
    than `reverse_rate_threshold` of sampled edges have their reverse present,
    we assume the edge list is directed-but-meant-undirected and add the
    flipped copies ourselves.
    """
    w_log = np.log1p(edges["weight"].astype(np.float32).to_numpy())
    w_norm = (w_log - w_log.min()) / (w_log.max() - w_log.min() + 1e-8)

    edges_proc = pd.DataFrame({
        "src_idx": edges["src"].map(id2idx).astype(np.int64),
        "dst_idx": edges["dst"].map(id2idx).astype(np.int64),
        "w_norm": w_norm.astype(np.float32),
    })

    if reverse_rate < reverse_rate_threshold:
        flipped = edges_proc.rename(columns={"src_idx": "dst_idx", "dst_idx": "src_idx"})
        edges_proc = pd.concat([edges_proc, flipped], ignore_index=True)

    edges_proc = edges_proc.groupby(["src_idx", "dst_idx"], as_index=False)["w_norm"].max()

    row = torch.from_numpy(edges_proc["src_idx"].to_numpy())
    col = torch.from_numpy(edges_proc["dst_idx"].to_numpy())
    edge_index = torch.stack([row, col], dim=0)
    edge_weight = torch.from_numpy(edges_proc["w_norm"].to_numpy())

    return Data(x=None, edge_index=edge_index, edge_attr=edge_weight)


def knn_edges(
    X: np.ndarray,
    k: int,
    mutual: bool,
) -> tuple[list[int], list[int], list[float]]:
    """Shared helper: cosine k-NN edges, optionally restricted to mutual
    neighbours, returned as (still one-directional) row/col/weight lists.
    Used by both the text-similarity graph and the fused-embedding graph
    built later for the final Louvain step."""
    n_neighbors = min(k + 1, len(X))
    try:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", n_jobs=-1)
    except TypeError:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)

    rows, cols, ws = [], [], []
    if mutual:
        adj_sets = [set(row[1:]) for row in indices]
        for i in range(len(X)):
            for j_idx, d in zip(indices[i][1:], distances[i][1:]):
                j = int(j_idx)
                if i in adj_sets[j]:
                    sim = 1.0 - float(d)
                    if sim > 0:
                        rows.append(i); cols.append(j); ws.append(sim)
    else:
        for i in range(len(X)):
            for j_idx, d in zip(indices[i][1:], distances[i][1:]):
                j = int(j_idx)
                sim = 1.0 - float(d)
                if sim > 0:
                    rows.append(i); cols.append(j); ws.append(sim)
    return rows, cols, ws


def build_text_knn_graph(X: np.ndarray, k: int, mutual: bool = True) -> Data:
    """Cosine (mutual) k-NN graph over the text/linguistic feature matrix."""
    rows, cols, ws = knn_edges(X, k, mutual)
    rows2 = rows + cols
    cols2 = cols + rows
    ws2 = ws + ws

    edge_index = torch.tensor([rows2, cols2], dtype=torch.long)
    edge_weight = torch.tensor(ws2, dtype=torch.float32)
    return Data(x=None, edge_index=edge_index, edge_attr=edge_weight)


def attach_node_features(data_struct: Data, data_text: Data, x: torch.Tensor) -> None:
    """Both streams see the exact same node feature matrix; only the graph
    topology differs between them."""
    data_struct.x = x.clone()
    data_text.x = x.clone()

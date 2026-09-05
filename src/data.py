"""
Loading and sanity-checking the raw Telegram dataset.

Expected inputs (see README.md for the exact schema):
  - `telegram_graph.edgelist`: whitespace-separated `src dst weight` triples,
     where `weight` is the number of shared members between two groups.
  - `graph_labels.csv`: columns `peerid, peer_name, about, label`.
    `label` is whatever reference/ground-truth grouping you evaluate
    NMI/ARI/Purity against -- document its provenance for your own dataset,
    the original notebook did not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    edges: pd.DataFrame          # columns: src, dst, weight (raw peerids)
    labels_df: pd.DataFrame      # columns: peerid, peer_name, about, label
    id2idx: dict
    idx2id: dict
    num_nodes: int


def load_dataset(edges_path: str, labels_path: str) -> Dataset:
    """Load the edge list and label table, restrict edges to known nodes,
    and build a dense 0..N-1 index mapping."""
    edges = pd.read_csv(edges_path, sep=" ", names=["src", "dst", "weight"])
    labels_df = pd.read_csv(labels_path)

    labels_df["peerid"] = labels_df["peerid"].astype(np.int64)
    edges = edges[edges["src"] != edges["dst"]].copy()
    edges["src"] = edges["src"].astype(np.int64)
    edges["dst"] = edges["dst"].astype(np.int64)
    edges["weight"] = edges["weight"].astype(np.int64)

    valid_ids = set(labels_df["peerid"].tolist())
    edges = edges[edges["src"].isin(valid_ids) & edges["dst"].isin(valid_ids)].copy()

    nodes_df = labels_df[["peerid", "peer_name", "about", "label"]].drop_duplicates().copy()
    id2idx = {pid: i for i, pid in enumerate(nodes_df["peerid"].tolist())}
    idx2id = {i: pid for pid, i in id2idx.items()}
    num_nodes = len(id2idx)

    return Dataset(
        edges=edges,
        labels_df=labels_df,
        id2idx=id2idx,
        idx2id=idx2id,
        num_nodes=num_nodes,
    )


def compute_reverse_rate(edges: pd.DataFrame, sample_n: int = 200_000, seed: int = 42) -> float:
    """Estimate what fraction of sampled (src, dst) pairs also appear as
    (dst, src) elsewhere in the edge list. Used to decide whether the edge
    list already stores both directions or needs to be symmetrized."""
    num_edges_raw = len(edges)
    n = min(sample_n, num_edges_raw)
    if n == 0:
        return 0.0
    sample = edges.sample(n=n, random_state=seed)
    m = sample.merge(
        edges[["src", "dst"]],
        how="left",
        left_on=["dst", "src"],
        right_on=["src", "dst"],
        indicator=True,
    )
    return float((m["_merge"] == "both").mean())


def edge_sanity_report(edges: pd.DataFrame, seed: int = 42) -> dict:
    """Compute the same diagnostics the original notebook printed:
    duplicate-edge ratio, approximate reverse-edge presence rate, and
    basic degree/weight statistics."""
    num_edges_raw = len(edges)
    dups = int(edges.duplicated(subset=["src", "dst"]).sum())
    dup_ratio = dups / max(1, num_edges_raw)
    reverse_rate = compute_reverse_rate(edges, seed=seed)

    deg_counts: dict = {}
    for u, v in zip(edges["src"].to_numpy(), edges["dst"].to_numpy()):
        deg_counts[u] = deg_counts.get(u, 0) + 1
        deg_counts[v] = deg_counts.get(v, 0) + 1
    deg_vals = np.array(list(deg_counts.values())) if deg_counts else np.array([0])
    w = edges["weight"].to_numpy()

    return {
        "num_edges_raw": num_edges_raw,
        "duplicate_count": dups,
        "duplicate_ratio": dup_ratio,
        "reverse_edge_rate": reverse_rate,
        "degree_mean": float(deg_vals.mean()),
        "degree_std": float(deg_vals.std()),
        "degree_min": int(deg_vals.min()),
        "degree_max": int(deg_vals.max()),
        "weight_mean": float(w.mean()) if len(w) else 0.0,
        "weight_std": float(w.std()) if len(w) else 0.0,
        "weight_min": float(w.min()) if len(w) else 0.0,
        "weight_max": float(w.max()) if len(w) else 0.0,
        "degree_values": deg_vals,  # kept for plotting; drop before json.dump
    }


def print_sanity_report(report: dict) -> None:
    print(f"Raw edges: {report['num_edges_raw']:,}")
    print(f"Duplicate (src,dst) count: {report['duplicate_count']:,} "
          f"(ratio {report['duplicate_ratio']:.4f})")
    print(f"Reverse-edge presence rate (approx): {report['reverse_edge_rate']:.3f}")
    print(f"Degree stats: mean={report['degree_mean']:.2f}, std={report['degree_std']:.2f}, "
          f"min={report['degree_min']}, max={report['degree_max']}")
    print(f"Weight stats: mean={report['weight_mean']:.2f}, std={report['weight_std']:.2f}, "
          f"min={report['weight_min']}, max={report['weight_max']}")

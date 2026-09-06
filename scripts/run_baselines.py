#!/usr/bin/env python
"""
Run classical community-detection baselines (Louvain, Label Propagation,
and -- if installed -- Leiden, Infomap) on the structural graph, each in a
weighted and unweighted variant, and evaluate against the reference
`label` column. Reproduces the Chapter 5 "comparison with classical
methods" table (NMI / ARI / Purity / Modularity / runtime per algorithm).

This only needs the structural graph -- no GATv2 training, no text
embeddings -- so it runs in seconds to minutes, not hours.

Usage:
    python -m scripts.run_baselines \\
        --edges data/telegram_graph.edgelist \\
        --labels data/graph_labels.csv \\
        --output-dir outputs

Leiden needs `pip install python-igraph leidenalg`; Infomap needs
`pip install infomap`. Missing ones are skipped with a printed note, not
a crash -- rerun after installing them to fill in the rest of the table.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_dataset, compute_reverse_rate
from src.graphs import build_structural_graph
from src.clustering import pyg_to_nx_weighted
from src.evaluate import build_label_index
from src.baselines import run_all_classical_baselines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--edges", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--resolution", type=float, default=1.0,
                    help="resolution parameter for Louvain/Leiden (default: 1.0)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--algorithms", nargs="+", default=None,
                    choices=["louvain", "label_propagation", "leiden", "infomap"],
                    help="subset of algorithms to run (default: all four)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Loading data & building structural graph ===")
    ds = load_dataset(args.edges, args.labels)
    reverse_rate = compute_reverse_rate(ds.edges, seed=args.seed)
    data_struct = build_structural_graph(ds.edges, ds.id2idx, reverse_rate)
    G = pyg_to_nx_weighted(data_struct.to("cpu"))
    print(f"Structural graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    y_true, _ = build_label_index(ds.labels_df, column="label")

    print("\n=== Running classical baselines ===")
    df = run_all_classical_baselines(
        G, y_true, resolution=args.resolution, seed=args.seed, algorithms=args.algorithms,
    )

    out_path = os.path.join(args.output_dir, "classical_baselines.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()

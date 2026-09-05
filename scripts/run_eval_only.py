#!/usr/bin/env python
"""
Re-cluster and re-evaluate from a previously saved `fused_embeddings.npy`
(from `run_pipeline.py`), without retraining the model. Handy for sweeping
`--k-fused` / `--louvain-resolution` cheaply.

Usage:
    python -m scripts.run_eval_only \\
        --labels data/graph_labels.csv \\
        --embeddings outputs/fused_embeddings.npy \\
        --output-dir outputs/resweep \\
        --k-fused 12 --louvain-resolution 1.2
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clustering import build_fused_knn_graph, run_louvain
from src.evaluate import eval_partition, build_label_index
from src import visualize as viz


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--output-dir", default="outputs/resweep")
    p.add_argument("--k-fused", type=int, default=8)
    p.add_argument("--mutual-fused", action="store_true", default=True)
    p.add_argument("--louvain-resolution", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plots", dest="save_plots", action="store_false", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    labels_df = pd.read_csv(args.labels)
    y_true, _ = build_label_index(labels_df, column="label")
    emb = np.load(args.embeddings)

    G = build_fused_knn_graph(emb, args.k_fused, mutual=args.mutual_fused)
    partition = run_louvain(G, resolution=args.louvain_resolution, seed=args.seed)
    Q, NMI, ARI, PUR, K, pred = eval_partition(G, partition, y_true)
    print(f"k={args.k_fused} res={args.louvain_resolution} -> "
          f"Q={Q:.4f} NMI={NMI:.4f} ARI={ARI:.4f} Purity={PUR:.4f} #communities={K}")

    if args.save_plots:
        viz.plot_community_sizes(
            pred, title=f"Community sizes (k={args.k_fused}, res={args.louvain_resolution})",
            save_path=os.path.join(args.output_dir, "community_sizes.png"),
        )

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"k_fused": args.k_fused, "resolution": args.louvain_resolution,
                    "Q": Q, "NMI": NMI, "ARI": ARI, "Purity": PUR, "num_communities": K},
                   f, indent=2)


if __name__ == "__main__":
    main()

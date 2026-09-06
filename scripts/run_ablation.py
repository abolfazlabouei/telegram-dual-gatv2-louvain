#!/usr/bin/env python
"""
Ablation studies for the Dual-GATv2 architecture. Every mode here
**retrains from scratch** (not a cheap post-hoc embedding trick), so the
numbers are safe to put in a paper/thesis table next to the main result.

Modes:
  no_align      -- full dual-stream model, but with lambda_alignment = 0
                   (answers: "does the cross-stream alignment loss help,
                   or does it just pull the two streams together for no
                   benefit?")
  struct_only   -- a single GATv2Encoder trained only on the structural
                   graph, clustered directly on its output (no fusion,
                   no content graph at all)
  content_only  -- the same, but only on the content (text-similarity)
                   graph
  all           -- runs all three back to back, freeing GPU memory
                   between each

All three share the exact same data loading, text features, and graph
construction as `scripts/run_pipeline.py`, and use the same default
hyperparameters (hidden dim, heads, dropout, epochs, patience, lr, k_fused,
resolution) unless overridden -- so the only thing that differs between a
row in `ablation_results.csv` and the `PROPOSED_fused` row in
`results_summary.csv` is the one architectural piece being ablated.

Usage:
    # one mode at a time
    python -m scripts.run_ablation --mode no_align \\
        --edges data/telegram_graph.edgelist --labels data/graph_labels.csv \\
        --output-dir outputs/ablation

    # all three, back to back
    python -m scripts.run_ablation --mode all \\
        --edges data/telegram_graph.edgelist --labels data/graph_labels.csv \\
        --output-dir outputs/ablation
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data import load_dataset, compute_reverse_rate
from src.text_features import (
    compute_text_embeddings, compute_linguistic_features, build_feature_matrix,
)
from src.graphs import build_structural_graph, build_text_knn_graph, attach_node_features
from src.models import DualGATv2, GATv2Encoder
from src.train import (
    set_seed, resolve_device, build_neighbor_loaders, train_dual_gatv2,
    build_single_neighbor_loader, train_single_stream_gatv2,
)
from src.embeddings import compute_all_embeddings, fuse_embeddings
from src.clustering import build_fused_knn_graph, run_louvain
from src.evaluate import build_label_index, collect_metrics_row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True,
                    choices=["no_align", "struct_only", "content_only", "all"])
    p.add_argument("--edges", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output-dir", default="outputs/ablation")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--k-fused", type=int, default=None)
    p.add_argument("--louvain-resolution", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    """Same defaults as `scripts/run_pipeline.py`'s Config -- only the
    handful of things this script exposes as flags are overridden, so
    every other hyperparameter matches the main run automatically."""
    cfg = Config()
    cfg.edges_path = args.edges
    cfg.labels_path = args.labels
    cfg.output_dir = args.output_dir
    for attr, val in [("epochs", args.epochs), ("batch_size", args.batch_size),
                      ("k_fused", args.k_fused), ("louvain_resolution", args.louvain_resolution),
                      ("seed", args.seed), ("device", args.device)]:
        if val is not None:
            setattr(cfg, attr, val)
    return cfg


def load_common(cfg: Config, device: torch.device):
    """Shared setup: dataset, text features, both graphs -- identical to
    the corresponding steps in `run_pipeline.py`, so every ablation is
    trained on exactly the same data as the main model."""
    ds = load_dataset(cfg.edges_path, cfg.labels_path)
    reverse_rate = compute_reverse_rate(ds.edges, seed=cfg.seed)

    X_text = compute_text_embeddings(
        ds.labels_df, cfg.sentence_model, cfg.emb_cache_path,
        device=str(device), batch_size=64,
    )
    ling = compute_linguistic_features(ds.labels_df)
    X = build_feature_matrix(X_text, ling, use_pca=cfg.use_pca, pca_dim=cfg.pca_dim, seed=cfg.seed)
    x = torch.tensor(X, dtype=torch.float32)

    data_struct = build_structural_graph(ds.edges, ds.id2idx, reverse_rate)
    data_text = build_text_knn_graph(X, cfg.k_text, mutual=cfg.use_mutual_text_knn)
    attach_node_features(data_struct, data_text, x)

    y_true, _ = build_label_index(ds.labels_df, column="label")
    return ds, data_struct.to("cpu"), data_text.to("cpu"), y_true, X.shape[1]


def cluster_and_evaluate(run_name: str, emb: np.ndarray, y_true: np.ndarray,
                          cfg: Config, extra_params: dict) -> tuple[dict, np.ndarray]:
    """Same fused-kNN-graph -> Louvain -> evaluate pipeline as the main
    run, applied to whatever embedding this ablation produced."""
    G = build_fused_knn_graph(emb, cfg.k_fused, mutual=cfg.use_mutual_fused)
    partition = run_louvain(G, resolution=cfg.louvain_resolution, seed=cfg.seed)
    pred = np.array([partition[i] for i in range(len(partition))], dtype=int)
    row = collect_metrics_row(run_name, G, partition, y_true, extra_params)
    return row, pred


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_no_align(cfg: Config, device, data_struct_cpu, data_text_cpu, y_true, in_dim, out_dir):
    print("\n=== Ablation: no_align (lambda_alignment = 0) ===")
    model = DualGATv2(in_dim, hidden=cfg.hidden_dim, out_dim=cfg.emb_out_dim,
                       heads=cfg.heads, dropout=cfg.dropout)
    loader_struct, loader_text = build_neighbor_loaders(
        data_struct_cpu, data_text_cpu, cfg.neighbors_struct, cfg.neighbors_text,
        cfg.batch_size, cfg.num_workers, cfg.shuffle_seeds,
    )
    model, history = train_dual_gatv2(
        model, loader_struct, loader_text, device,
        epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
        lambda_s=cfg.lambda_s, lambda_t=cfg.lambda_t,
        lambda_a=0.0,  # <-- the actual ablation: alignment loss switched off
        patience=cfg.patience, recon_num_samples=cfg.recon_num_samples,
    )
    del loader_struct, loader_text
    free_gpu()

    hs_all = compute_all_embeddings(model.enc_s, data_struct_cpu, cfg.neighbors_struct,
                                     device, cfg.batch_size, cfg.num_workers)
    ht_all = compute_all_embeddings(model.enc_t, data_text_cpu, cfg.neighbors_text,
                                     device, cfg.batch_size, cfg.num_workers)
    emb, alpha = fuse_embeddings(model, hs_all, ht_all)
    np.save(os.path.join(out_dir, "no_align_embeddings.npy"), emb)
    np.save(os.path.join(out_dir, "no_align_alpha.npy"), alpha)

    row, pred = cluster_and_evaluate(
        "ABLATION_no_align", emb, y_true, cfg,
        {"k_fused": cfg.k_fused, "resolution": cfg.louvain_resolution,
         "lambda_a": 0.0, "seed": cfg.seed},
    )
    return row, pred, history


def run_single_stream(cfg: Config, device, data_cpu, y_true, in_dim, stream_name, out_dir):
    print(f"\n=== Ablation: {stream_name}_only ===")
    encoder = GATv2Encoder(in_dim, hidden=cfg.hidden_dim, out_dim=cfg.emb_out_dim,
                            heads=cfg.heads, dropout=cfg.dropout)
    neighbors = cfg.neighbors_struct if stream_name == "struct" else cfg.neighbors_text
    loader = build_single_neighbor_loader(
        data_cpu, neighbors, cfg.batch_size, cfg.num_workers, cfg.shuffle_seeds,
    )
    encoder, history = train_single_stream_gatv2(
        encoder, loader, device, epochs=cfg.epochs, lr=cfg.lr,
        weight_decay=cfg.weight_decay, patience=cfg.patience,
        recon_num_samples=cfg.recon_num_samples,
    )
    del loader
    free_gpu()

    h_all = compute_all_embeddings(encoder, data_cpu, neighbors, device,
                                    cfg.batch_size, cfg.num_workers)
    emb = h_all.numpy()
    np.save(os.path.join(out_dir, f"{stream_name}_only_embeddings.npy"), emb)

    row, pred = cluster_and_evaluate(
        f"ABLATION_{stream_name}_only", emb, y_true, cfg,
        {"k_fused": cfg.k_fused, "resolution": cfg.louvain_resolution, "seed": cfg.seed},
    )
    return row, pred, history


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    os.makedirs(cfg.output_dir, exist_ok=True)

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"[run_ablation] device = {device}")

    print("=== Loading data & building both graphs (shared across all modes) ===")
    ds, data_struct_cpu, data_text_cpu, y_true, in_dim = load_common(cfg, device)

    modes = ["no_align", "struct_only", "content_only"] if args.mode == "all" else [args.mode]
    results = []

    for mode in modes:
        set_seed(cfg.seed)  # reset before each run: every ablation starts from the same seed

        if mode == "no_align":
            row, pred, history = run_no_align(
                cfg, device, data_struct_cpu, data_text_cpu, y_true, in_dim, cfg.output_dir)
        elif mode == "struct_only":
            row, pred, history = run_single_stream(
                cfg, device, data_struct_cpu, y_true, in_dim, "struct", cfg.output_dir)
        elif mode == "content_only":
            row, pred, history = run_single_stream(
                cfg, device, data_text_cpu, y_true, in_dim, "content", cfg.output_dir)
        else:
            raise ValueError(f"unknown mode: {mode}")

        print(f"[ABLATION {mode}] Q={row['Q']:.4f} NMI={row['NMI']:.4f} "
              f"ARI={row['ARI']:.4f} Purity={row['Purity']:.4f} #C={row['#C']}")

        pd.DataFrame({"epoch": range(1, len(history) + 1), "loss": history}).to_csv(
            os.path.join(cfg.output_dir, f"{mode}_training_loss.csv"), index=False,
        )
        pd.DataFrame({
            "peerid": [ds.idx2id[i] for i in range(len(pred))], "community": pred,
        }).to_csv(os.path.join(cfg.output_dir, f"{mode}_communities.csv"), index=False)

        results.append(row)
        free_gpu()

    results_df = pd.DataFrame(results)
    out_path = os.path.join(cfg.output_dir, "ablation_results.csv")
    # append to an existing file from a previous partial run instead of overwriting,
    # so `--mode X` can be called multiple times (once per mode) and still end up
    # with one combined table.
    if os.path.exists(out_path):
        prior = pd.read_csv(out_path)
        results_df = pd.concat([prior[~prior["run_name"].isin(results_df["run_name"])],
                                 results_df], ignore_index=True)
    results_df.to_csv(out_path, index=False)

    print(f"\nSaved -> {out_path}")
    print(results_df.to_string(index=False))
    print("\nColumns match `results_summary.csv` (run_name, Q, #C, NMI, ARI, Purity, ...) "
          "-- concatenate the two files for the full comparison table.")


if __name__ == "__main__":
    main()

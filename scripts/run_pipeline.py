#!/usr/bin/env python
"""
End-to-end pipeline:

  load data -> graph stats -> text features -> build graphs ->
  train Dual-GATv2 -> fuse embeddings -> Louvain on fused kNN graph ->
  evaluate -> (optional) k-sensitivity sweep -> (optional) structural-only
  baseline -> save everything to `--output-dir`.

Usage:
    python -m scripts.run_pipeline \\
        --edges data/telegram_graph.edgelist \\
        --labels data/graph_labels.csv \\
        --output-dir outputs

Run from the repository root so the `src` package resolves; or
`pip install -e .` first. See README.md for the full option list.
"""

from __future__ import annotations

# Headless-safe: set the backend before anything imports pyplot.
import matplotlib
matplotlib.use("Agg")

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

# Allow running as `python scripts/run_pipeline.py` from the repo root
# without having installed the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, config_from_args
from src.data import load_dataset, edge_sanity_report, print_sanity_report
from src.text_features import (
    compute_text_embeddings, compute_linguistic_features, build_feature_matrix,
)
from src.graphs import build_structural_graph, build_text_knn_graph, attach_node_features
from src.models import DualGATv2
from src.train import set_seed, resolve_device, build_neighbor_loaders, train_dual_gatv2
from src.embeddings import compute_all_embeddings, fuse_embeddings
from src.clustering import (
    build_fused_knn_graph, run_louvain, structural_only_baseline, k_sensitivity_sweep,
    pyg_to_nx_weighted,
)
from src.evaluate import eval_partition, build_label_index, collect_metrics_row
from src.reporting import graph_report, save_community_report, save_alpha_report
from src import visualize as viz


def main(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"[run_pipeline] device = {device}")

    fig_dir = os.path.join(cfg.output_dir, "figures")
    data_dir = os.path.join(cfg.output_dir, "data_for_figures")
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # ---------------------------------------------------------------- 1. data
    print("\n=== 1) Loading data ===")
    ds = load_dataset(cfg.edges_path, cfg.labels_path)
    print(f"Edges: {ds.edges.shape}, Labels: {ds.labels_df.shape}, Nodes: {ds.num_nodes}")

    # Quick raw-edge diagnostics (duplicate ratio, reverse-edge rate) --
    # printed only; the authoritative degree/weight/component figures
    # come from graph_report() on the built graphs, below.
    report = edge_sanity_report(ds.edges, seed=cfg.seed)
    print_sanity_report(report)
    reverse_rate = report["reverse_edge_rate"]

    # ---------------------------------------------------------- 2. text features
    print("\n=== 2) Text features ===")
    X_text = compute_text_embeddings(
        ds.labels_df, cfg.sentence_model, cfg.emb_cache_path,
        device=str(device), batch_size=64,
    )
    ling = compute_linguistic_features(ds.labels_df)
    X = build_feature_matrix(X_text, ling, use_pca=cfg.use_pca, pca_dim=cfg.pca_dim, seed=cfg.seed)
    x = torch.tensor(X, dtype=torch.float32)
    print(f"Feature matrix X: {X.shape}")

    # -------------------------------------------------------------- 3. graphs
    print("\n=== 3) Building graphs ===")
    data_struct = build_structural_graph(ds.edges, ds.id2idx, reverse_rate)
    print(f"Structural edges: {data_struct.edge_index.shape}")

    data_text = build_text_knn_graph(X, cfg.k_text, mutual=cfg.use_mutual_text_knn)
    print(f"Text-sim edges: {data_text.edge_index.shape[1]}")

    attach_node_features(data_struct, data_text, x)
    data_struct_cpu, data_text_cpu = data_struct.to("cpu"), data_text.to("cpu")

    y_true, _ = build_label_index(ds.labels_df, column="label")

    # Full figure + CSV report for both input graphs (degree/weight
    # distributions, degree CCDF, rank-size, clustering-vs-degree,
    # k-core curve, connected components, one-row summary table --
    # matches the "graph statistics" tables/figures in Chapter 5).
    print("\n--- Structural graph report ---")
    G_struct_nx = pyg_to_nx_weighted(data_struct_cpu)
    struct_stats = graph_report(
        G_struct_nx, "structural", fig_dir, data_dir,
        weighted=True, run_power_law=cfg.run_power_law_fit, save_plots=cfg.save_plots,
    )
    print(struct_stats)

    print("\n--- Content (text-similarity) graph report ---")
    G_text_nx = pyg_to_nx_weighted(data_text_cpu)
    content_stats = graph_report(
        G_text_nx, "content", fig_dir, data_dir,
        weighted=True, run_power_law=False, save_plots=cfg.save_plots,
    )
    print(content_stats)

    # ------------------------------------------------------------- 4. model
    print("\n=== 4) Training Dual-GATv2 ===")
    in_dim = int(data_struct_cpu.x.size(1))
    model = DualGATv2(in_dim, hidden=cfg.hidden_dim, out_dim=cfg.emb_out_dim,
                       heads=cfg.heads, dropout=cfg.dropout)

    loader_struct, loader_text = build_neighbor_loaders(
        data_struct_cpu, data_text_cpu,
        cfg.neighbors_struct, cfg.neighbors_text,
        cfg.batch_size, cfg.num_workers, cfg.shuffle_seeds,
    )
    model, history = train_dual_gatv2(
        model, loader_struct, loader_text, device,
        epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
        lambda_s=cfg.lambda_s, lambda_t=cfg.lambda_t, lambda_a=cfg.lambda_a,
        patience=cfg.patience, recon_num_samples=cfg.recon_num_samples,
    )
    pd.DataFrame({"epoch": range(1, len(history) + 1), "loss": history}).to_csv(
        os.path.join(cfg.output_dir, "training_loss.csv"), index=False,
    )

    # -------------------------------------------------------- 5. embeddings
    print("\n=== 5) Computing & fusing embeddings ===")
    hs_all = compute_all_embeddings(model.enc_s, data_struct_cpu, cfg.neighbors_struct,
                                     device, cfg.batch_size, cfg.num_workers)
    ht_all = compute_all_embeddings(model.enc_t, data_text_cpu, cfg.neighbors_text,
                                     device, cfg.batch_size, cfg.num_workers)
    emb, alpha = fuse_embeddings(model, hs_all, ht_all)
    np.save(os.path.join(cfg.output_dir, "fused_embeddings.npy"), emb)
    np.save(os.path.join(cfg.output_dir, "gate_alpha.npy"), alpha)
    save_alpha_report(alpha, ds.idx2id, fig_dir, data_dir, save_plots=cfg.save_plots)

    # Free everything only needed up through embedding extraction. G_struct_nx
    # is kept -- reused as-is for the structural-only baseline below, instead
    # of rebuilding a second full copy of a multi-million-edge graph while
    # the first is still resident (that duplication was causing an OOM kill
    # at this step).
    del data_struct_cpu, data_text_cpu, hs_all, ht_all, G_text_nx, loader_struct, loader_text
    import gc
    gc.collect()

    # -------------------------------------------------------- 6. clustering
    print("\n=== 6) Fused kNN graph + Louvain ===")
    G = build_fused_knn_graph(emb, cfg.k_fused, mutual=cfg.use_mutual_fused)
    partition = run_louvain(G, resolution=cfg.louvain_resolution, seed=cfg.seed)
    Q, NMI, ARI, PUR, K, pred = eval_partition(G, partition, y_true)
    print(f"[PROPOSED fused-kNN + Louvain] Q={Q:.4f} NMI={NMI:.4f} ARI={ARI:.4f} "
          f"Purity={PUR:.4f} #communities={K}")

    print("\n--- Fused kNN graph report ---")
    fused_stats = graph_report(
        G, "fused", fig_dir, data_dir,
        weighted=True, run_power_law=False, save_plots=cfg.save_plots,
    )
    print(fused_stats)
    save_community_report(pred, fig_dir, data_dir, name="fused", save_plots=cfg.save_plots)

    pd.DataFrame({"peerid": [ds.idx2id[i] for i in range(len(pred))], "community": pred}).to_csv(
        os.path.join(cfg.output_dir, "communities_fused.csv"), index=False,
    )

    results = [collect_metrics_row("PROPOSED_fused", G, partition, y_true, {
        "k_fused": cfg.k_fused, "mutual_fused": cfg.use_mutual_fused,
        "resolution": cfg.louvain_resolution, "lambda_a": cfg.lambda_a,
        "k_text": cfg.k_text, "mutual_text": cfg.use_mutual_text_knn,
        "neighbors_struct": str(cfg.neighbors_struct), "neighbors_text": str(cfg.neighbors_text),
        "heads": cfg.heads, "hidden": cfg.hidden_dim, "seed": cfg.seed,
    })]

    # ---------------------------------------------------- 7. optional extras
    if cfg.run_k_sensitivity:
        print("\n=== 7) k-sensitivity sweep ===")
        sweep_df = k_sensitivity_sweep(
            emb, y_true, cfg.k_sensitivity_list, cfg.use_mutual_fused,
            cfg.louvain_resolution, cfg.seed,
        )
        print(sweep_df.to_string(index=False))
        sweep_df.to_csv(os.path.join(cfg.output_dir, "k_sensitivity.csv"), index=False)
        if cfg.save_plots:
            viz.plot_k_sensitivity(sweep_df, save_path=os.path.join(fig_dir, "k_sensitivity.png"))

    if cfg.run_struct_only_baseline:
        print("\n=== 8) Structural-only Louvain baseline ===")
        base = structural_only_baseline(G_struct_nx, y_true,
                                         resolution=cfg.louvain_resolution, seed=cfg.seed)
        print(f"[STRUCT-only Louvain] Q={base['Q']:.4f} NMI={base['NMI']:.4f} "
              f"ARI={base['ARI']:.4f} Purity={base['Purity']:.4f} "
              f"#communities={base['num_communities']}")
        results.append({
            "run_name": "STRUCT_only_louvain", "Q": base["Q"], "#C": base["num_communities"],
            "NMI": base["NMI"], "ARI": base["ARI"], "Purity": base["Purity"],
            "resolution": cfg.louvain_resolution, "seed": cfg.seed,
        })

    # -------------------------------------------------------------- 9. save
    pd.DataFrame([
        {"graph": "structural", **struct_stats},
        {"graph": "content", **content_stats},
        {"graph": "fused", **fused_stats},
    ]).to_csv(os.path.join(data_dir, "all_graph_summaries.csv"), index=False)

    results_df = pd.DataFrame(results)
    results_path = os.path.join(cfg.output_dir, "results_summary.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results summary to {results_path}")

    with open(os.path.join(cfg.output_dir, "config_used.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)

    print("\nDone.")


if __name__ == "__main__":
    main(config_from_args())

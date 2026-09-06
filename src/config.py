"""
Central configuration for the pipeline.

Every hyperparameter that used to live as a bare global variable in the
original notebook is collected here as a single `Config` dataclass, with
sane defaults matching the original run. Override any of them from the
command line via `scripts/run_pipeline.py --help`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # ---- Paths -----------------------------------------------------
    edges_path: str = "data/telegram_graph.edgelist"
    labels_path: str = "data/graph_labels.csv"
    output_dir: str = "outputs"
    emb_cache_path: str = "outputs/text_embeds.npy"

    # ---- Text embedding model --------------------------------------
    sentence_model: str = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    use_pca: bool = False
    pca_dim: int = 128

    # ---- Text similarity kNN graph -----------------------------------
    k_text: int = 8
    use_mutual_text_knn: bool = True

    # ---- GATv2 encoder architecture ----------------------------------
    hidden_dim: int = 64
    emb_out_dim: int = 32
    heads: int = 2
    dropout: float = 0.10

    # ---- Training -----------------------------------------------------
    lr: float = 7e-4
    weight_decay: float = 5e-5
    epochs: int = 50
    patience: int = 10
    lambda_s: float = 1.0      # weight of structural-stream reconstruction loss
    lambda_t: float = 1.0      # weight of text-stream reconstruction loss
    lambda_a: float = 0.05     # weight of the cross-stream alignment loss
    recon_num_samples: int = 20000

    # ---- NeighborLoader mini-batching ----------------------------------
    neighbors_struct: list = field(default_factory=lambda: [15, 10])
    neighbors_text: list = field(default_factory=lambda: [15, 10])
    batch_size: int = 4096
    num_workers: int = 4
    shuffle_seeds: bool = False

    # ---- Final clustering on fused embeddings --------------------------
    k_fused: int = 8
    use_mutual_fused: bool = True
    louvain_resolution: float = 1.0

    # ---- Misc -----------------------------------------------------------
    seed: int = 42
    device: str = "auto"  # "auto" | "cuda" | "cpu"

    # ---- Optional analysis toggles ---------------------------------------
    run_k_sensitivity: bool = True
    k_sensitivity_list: list = field(default_factory=lambda: [5, 8, 10, 15, 20])
    run_struct_only_baseline: bool = True
    run_power_law_fit: bool = False  # needs the optional `powerlaw` package
    save_plots: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI overrides for the most commonly-changed settings.

    Anything not exposed here can still be changed by editing this file's
    defaults or by constructing `Config(...)` directly in your own script.
    """
    p = argparse.ArgumentParser(
        description="Dual-stream GATv2 + Louvain community detection on Telegram groups."
    )
    p.add_argument("--edges", dest="edges_path", type=str, default=None)
    p.add_argument("--labels", dest="labels_path", type=str, default=None)
    p.add_argument("--output-dir", dest="output_dir", type=str, default=None)
    p.add_argument("--emb-cache-path", dest="emb_cache_path", type=str, default=None)
    p.add_argument("--sentence-model", dest="sentence_model", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--k-text", dest="k_text", type=int, default=None)
    p.add_argument("--k-fused", dest="k_fused", type=int, default=None)
    p.add_argument("--louvain-resolution", dest="louvain_resolution", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"])
    p.add_argument("--no-k-sensitivity", dest="run_k_sensitivity", action="store_false", default=None)
    p.add_argument("--no-struct-baseline", dest="run_struct_only_baseline", action="store_false", default=None)
    p.add_argument("--power-law-fit", dest="run_power_law_fit", action="store_true", default=None)
    p.add_argument("--no-plots", dest="save_plots", action="store_false", default=None)
    return p


def config_from_args(argv: list[str] | None = None) -> Config:
    """Build a Config, starting from defaults and applying any CLI overrides."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = Config()
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg

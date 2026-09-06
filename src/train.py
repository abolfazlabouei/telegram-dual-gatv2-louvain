"""
Mini-batch training of DualGATv2 with aligned `NeighborLoader`s over the
structural and text-similarity graphs, mixed precision, and early stopping
on the (unsupervised) training loss.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from .losses import recon_loss, alignment_loss
from .models import DualGATv2


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_neighbor_loaders(
    data_struct: Data,
    data_text: Data,
    neighbors_struct: list[int],
    neighbors_text: list[int],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> tuple[NeighborLoader, NeighborLoader]:
    """Two loaders sharing the same seed-node ordering per batch (enforced
    by `shuffle=False` on both, or by passing the same `torch.manual_seed`
    upstream if you do want shuffling)."""
    num_nodes = data_struct.num_nodes
    input_nodes = torch.arange(num_nodes)

    loader_struct = NeighborLoader(
        data_struct, num_neighbors=neighbors_struct,
        input_nodes=input_nodes, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers,
        subgraph_type="directional",
    )
    loader_text = NeighborLoader(
        data_text, num_neighbors=neighbors_text,
        input_nodes=input_nodes, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers,
        subgraph_type="directional",
    )
    return loader_struct, loader_text


def build_single_neighbor_loader(
    data: Data,
    neighbors: list[int],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> NeighborLoader:
    """One `NeighborLoader` over a single graph -- used for the single-
    stream ablations (structural-only / content-only), where there is no
    second stream to keep aligned."""
    input_nodes = torch.arange(data.num_nodes)
    return NeighborLoader(
        data, num_neighbors=neighbors,
        input_nodes=input_nodes, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers,
        subgraph_type="directional",
    )


def train_single_stream_gatv2(
    encoder: torch.nn.Module,
    loader: NeighborLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    recon_num_samples: int = 20000,
    verbose: bool = True,
) -> tuple[torch.nn.Module, list[float]]:
    """Train a single `GATv2Encoder` on its own reconstruction loss only --
    the single-stream counterpart of `train_dual_gatv2`, used for the
    structural-only / content-only ablations (no fusion, no alignment
    loss, since there is only one stream).
    """
    encoder = encoder.to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best, wait, best_state = float("inf"), 0, None
    history: list[float] = []

    for ep in range(1, epochs + 1):
        encoder.train()
        loss_meter = []

        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                h_full = encoder(batch.x, batch.edge_index, batch.edge_attr)
                loss = recon_loss(h_full, batch, num_samples=recon_num_samples)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_meter.append(loss.item())

        epoch_loss = float(np.mean(loss_meter)) if loss_meter else float("nan")
        history.append(epoch_loss)

        if epoch_loss < best - 1e-4:
            best, wait = epoch_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
        else:
            wait += 1

        if verbose:
            print(f"[EP {ep:03d}] loss={epoch_loss:.4f} wait={wait}")

        if wait >= patience:
            if verbose:
                print("Early stopping.")
            break

    if best_state is not None:
        encoder.load_state_dict(best_state)

    return encoder, history


def train_dual_gatv2(
    model: DualGATv2,
    loader_struct: NeighborLoader,
    loader_text: NeighborLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    lambda_s: float,
    lambda_t: float,
    lambda_a: float,
    patience: int,
    recon_num_samples: int = 20000,
    verbose: bool = True,
) -> tuple[DualGATv2, list[float]]:
    """Train in place and return (model with best-loss weights loaded,
    per-epoch training loss history)."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best, wait, best_state = float("inf"), 0, None
    history: list[float] = []

    for ep in range(1, epochs + 1):
        model.train()
        loss_meter = []

        for batch_s, batch_t in zip(loader_struct, loader_text):
            bs = int(batch_s.batch_size)
            bt = int(batch_t.batch_size)
            assert bs == bt, f"Seed sizes differ: {bs} vs {bt}"
            assert torch.equal(batch_s.n_id[:bs], batch_t.n_id[:bt]), \
                "Seed mismatch between streams -- structural and text graphs must " \
                "share the same node indexing (see graphs.attach_node_features)."

            batch_s = batch_s.to(device)
            batch_t = batch_t.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                hs_full = model.enc_s(batch_s.x, batch_s.edge_index, batch_s.edge_attr)
                ht_full = model.enc_t(batch_t.x, batch_t.edge_index, batch_t.edge_attr)

                hs_seed = hs_full[:bs, :]
                ht_seed = ht_full[:bt, :]

                ls = recon_loss(hs_full, batch_s, num_samples=recon_num_samples)
                lt = recon_loss(ht_full, batch_t, num_samples=recon_num_samples)
                la = alignment_loss(hs_seed, ht_seed)

                loss = lambda_s * ls + lambda_t * lt + lambda_a * la

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_meter.append(loss.item())

        epoch_loss = float(np.mean(loss_meter)) if loss_meter else float("nan")
        history.append(epoch_loss)

        if epoch_loss < best - 1e-4:
            best, wait = epoch_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if verbose:
            print(f"[EP {ep:03d}] loss={epoch_loss:.4f} wait={wait}")

        if wait >= patience:
            if verbose:
                print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history

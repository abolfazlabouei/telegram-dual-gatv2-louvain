"""
Run the trained encoders over every node (not just mini-batches) to get
final per-node embeddings, then fuse the two streams with the learned
gate.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from .models import DualGATv2


@torch.no_grad()
def compute_all_embeddings(
    encoder: torch.nn.Module,
    data: Data,
    num_neighbors: list[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    """Full-graph inference via mini-batches, written back into a single
    (num_nodes, out_dim) tensor indexed by the original node ids."""
    out_dim = encoder.conv2.out_channels
    out_tensor = torch.empty((data.num_nodes, out_dim), dtype=torch.float32, device="cpu")

    eval_loader = NeighborLoader(
        data, num_neighbors=num_neighbors,
        input_nodes=torch.arange(data.num_nodes),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        subgraph_type="directional",
    )
    encoder.eval()
    for batch in eval_loader:
        n_id = batch.n_id
        batch = batch.to(device)
        h = encoder(batch.x, batch.edge_index, batch.edge_attr)
        out_tensor[n_id] = h.detach().cpu()

    return out_tensor


def fuse_embeddings(model: DualGATv2, hs_all: torch.Tensor, ht_all: torch.Tensor
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Apply the trained fusion gate on CPU over the full node set.
    Returns (fused_embeddings, alpha) as numpy arrays."""
    with torch.no_grad():
        model.fuse.to("cpu")
        model.fuse.eval()
        z = torch.cat([hs_all, ht_all], dim=-1)
        alpha = torch.sigmoid(model.fuse.mlp(z)).squeeze(-1)
        hf_all = alpha.unsqueeze(-1) * hs_all + (1 - alpha).unsqueeze(-1) * ht_all

    return hf_all.numpy(), alpha.numpy()

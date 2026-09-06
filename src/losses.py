"""
Self-supervised training objective for Dual-GATv2:

  total_loss = lambda_s * recon_loss(h_struct)
             + lambda_t * recon_loss(h_text)
             + lambda_a * alignment_loss(h_struct_seed, h_text_seed)

`recon_loss` is a standard dot-product-decoder link-prediction loss with
random negative sampling (as in a Graph Autoencoder). `alignment_loss` is
an MSE term that pulls the two streams' seed-node embeddings towards each
other -- note that this works *against* the goal of keeping the two
streams complementary; keep an eye on `lambda_a` and consider an ablation
with it set to 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

bce_logits = nn.BCEWithLogitsLoss()


def sample_edges(edge_index: torch.Tensor, num_nodes: int, num_samples: int = 20000,
                  device: str | torch.device = "cpu"):
    """Sample `num_samples` positive edges and an equal number of random
    (likely-negative) node pairs."""
    E = edge_index.size(1)
    if E == 0:
        return None
    idx = torch.randint(0, E, (min(num_samples, E),), device=device)
    pos = edge_index[:, idx]
    neg = torch.randint(0, num_nodes, pos.size(), device=device)
    return pos, neg


def dot_decode(h: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    src, dst = edges[0], edges[1]
    return (h[src] * h[dst]).sum(dim=-1)


def recon_loss(h: torch.Tensor, data, num_samples: int = 20000) -> torch.Tensor:
    """Binary cross-entropy between real edges (label 1) and random node
    pairs (label 0), scored by the dot product of learned embeddings."""
    res = sample_edges(data.edge_index, data.num_nodes, num_samples=num_samples, device=h.device)
    if res is None:
        return torch.tensor(0.0, device=h.device)
    pos, neg = res
    pos_score = dot_decode(h, pos)
    neg_score = dot_decode(h, neg)
    y_pos = torch.ones_like(pos_score)
    y_neg = torch.zeros_like(neg_score)
    return bce_logits(pos_score, y_pos) + bce_logits(neg_score, y_neg)


def alignment_loss(hs: torch.Tensor, ht: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(hs, ht)

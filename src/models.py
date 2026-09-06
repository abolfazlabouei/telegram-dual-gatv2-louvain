"""
Dual-stream GATv2 architecture:

  - `GATv2Encoder`: a 2-layer GATv2 encoder applied independently to the
    structural graph and to the text-similarity graph.
  - `FusionGate`: a small MLP that reads both streams' embeddings for a
    node and outputs a per-node scalar alpha in [0, 1], used to blend the
    two embeddings as `alpha * h_struct + (1 - alpha) * h_text`.
  - `DualGATv2`: bundles two encoders and one fusion gate.

Note: this is a *convex combination* fusion (matches the actual training
code), not the concatenation-then-linear-layer fusion described in some
drafts of the thesis text -- make sure your written Chapter 4 matches
whichever one you actually train with.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GATv2Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 32,
                 heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden, heads=heads, dropout=dropout, edge_dim=1)
        self.conv2 = GATv2Conv(hidden * heads, out_dim, heads=1, dropout=dropout,
                                edge_dim=1, concat=False)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index, edge_attr.unsqueeze(-1))
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h, edge_index, edge_attr.unsqueeze(-1))
        return h


class FusionGate(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, hs: torch.Tensor, ht: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([hs, ht], dim=-1)
        alpha = torch.sigmoid(self.mlp(z))
        hf = alpha * hs + (1 - alpha) * ht
        return hf, alpha.squeeze(-1)


class DualGATv2(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 32,
                 heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.enc_s = GATv2Encoder(in_dim, hidden, out_dim, heads, dropout)
        self.enc_t = GATv2Encoder(in_dim, hidden, out_dim, heads, dropout)
        self.fuse = FusionGate(out_dim)

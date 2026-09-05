"""
Turn two raw input files into the `telegram_graph.edgelist` +
`graph_labels.csv` pair that `scripts/run_pipeline.py` expects.

You need two raw files:

  1. a **membership table** -- one row per (user, group) pair: which
     numeric user id belongs to which numeric group id. This is the
     bipartite user-group graph the structural edges get computed from.
  2. a **group metadata table** -- one row per group: its numeric id, a
     text description ("about"), and a reference label. A display-name
     column is picked up too if present, otherwise the group id is used
     as a placeholder name.

Column names almost never match by luck across two different raw
exports, so every column name below is a parameter -- pass the matching
`--*-col` flag in `scripts/prepare_data.py` if your files use different
headers than the defaults.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _require_columns(df: pd.DataFrame, cols: list[str], path: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing column(s) {missing}. Found columns: {list(df.columns)}. "
            f"Pass the matching --*-col flag to point at your real column name(s)."
        )


def load_membership_table(path: str, user_col: str = "user_id",
                           group_col: str = "group_id") -> pd.DataFrame:
    """One row per (user, group) membership. Drops rows with a missing id
    on either side and exact duplicate rows."""
    df = pd.read_csv(path)
    _require_columns(df, [user_col, group_col], path)
    df = df[[user_col, group_col]].rename(columns={user_col: "user_id", group_col: "group_id"})

    before = len(df)
    df = df.dropna().drop_duplicates()
    df["user_id"] = df["user_id"].astype(np.int64)
    df["group_id"] = df["group_id"].astype(np.int64)
    print(f"[preprocessing] memberships: {before:,} raw rows -> {len(df):,} after dropna + de-dup")
    return df


def load_group_metadata(path: str, group_col: str = "peerid", about_col: str = "about",
                         label_col: str = "label", name_col: str = "peer_name") -> pd.DataFrame:
    """One row per group, standardized to columns peerid/peer_name/about/label.
    If `name_col` isn't present, the group id is used as a placeholder name
    (so downstream text embedding still has *something* to encode)."""
    df = pd.read_csv(path)
    _require_columns(df, [group_col, about_col, label_col], path)

    rename = {group_col: "peerid", about_col: "about", label_col: "label"}
    keep = [group_col, about_col, label_col]
    if name_col in df.columns:
        rename[name_col] = "peer_name"
        keep.append(name_col)

    df = df[keep].rename(columns=rename)
    if "peer_name" not in df.columns:
        df["peer_name"] = df["peerid"].astype(str)
        print(f"[preprocessing] no '{name_col}' column found; using the group id "
              f"as a placeholder peer_name")

    before = len(df)
    df["peerid"] = df["peerid"].astype(np.int64)
    df["about"] = df["about"].fillna("")
    df["label"] = df["label"].fillna("unknown")
    df = df.drop_duplicates(subset="peerid")
    print(f"[preprocessing] group metadata: {before:,} raw rows -> {len(df):,} after de-dup by id")
    return df[["peerid", "peer_name", "about", "label"]]


def compute_shared_membership_edges(memberships: pd.DataFrame, min_shared: int = 5) -> pd.DataFrame:
    """Bipartite projection: for every pair of groups, count how many
    users belong to both, keeping only pairs with `weight >= min_shared`.

    Implemented as one sparse matrix product (M^T @ M) rather than a
    nested Python loop, since a naive per-pair comparison does not scale
    past a few thousand groups -- this scales to millions of membership
    rows and tens of thousands of groups.
    """
    user_ids = memberships["user_id"].unique()
    group_ids = memberships["group_id"].unique()
    user2idx = {u: i for i, u in enumerate(user_ids)}
    group2idx = {g: i for i, g in enumerate(group_ids)}

    rows = memberships["user_id"].map(user2idx).to_numpy()
    cols = memberships["group_id"].map(group2idx).to_numpy()
    data = np.ones(len(memberships), dtype=np.int32)

    M = sp.csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(group_ids)))
    print(f"[preprocessing] membership matrix: {M.shape[0]:,} users x {M.shape[1]:,} groups, "
          f"{M.nnz:,} nonzero entries")

    # Group-group shared-member counts. The diagonal of C is each group's
    # own size (a group always fully "shares" with itself) -- we only
    # want the off-diagonal, upper-triangle part as edges.
    C = (M.T @ M).tocoo()
    print(f"[preprocessing] co-membership matrix computed ({C.nnz:,} nonzero group pairs, "
          f"including the diagonal)")

    mask = (C.row < C.col) & (C.data >= min_shared)
    src_idx = C.row[mask]
    dst_idx = C.col[mask]
    weight = C.data[mask]

    idx2group = {i: g for g, i in group2idx.items()}
    edges = pd.DataFrame({
        "src": [idx2group[i] for i in src_idx],
        "dst": [idx2group[i] for i in dst_idx],
        "weight": weight,
    })
    print(f"[preprocessing] {len(edges):,} edges with >= {min_shared} shared members "
          f"(out of {len(group_ids):,} groups seen in the membership table)")
    return edges


def restrict_to_common_nodes(edges: pd.DataFrame, labels_df: pd.DataFrame
                              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only groups present in BOTH files: a group with no metadata
    can't be text-embedded or evaluated, and a group with no edges is an
    isolated node the structural graph gets no signal from anyway.

    This is the "shared node" step -- run it once here so `src/data.py`'s
    own (redundant, cheap) filtering downstream is just a safety net, not
    where most of the dropping happens.
    """
    edge_ids = set(edges["src"]).union(edges["dst"])
    meta_ids = set(labels_df["peerid"])
    common = edge_ids & meta_ids

    print(f"[preprocessing] groups with >=1 edge: {len(edge_ids):,} | "
          f"groups with metadata: {len(meta_ids):,} | common to both: {len(common):,}")

    edges = edges[edges["src"].isin(common) & edges["dst"].isin(common)].copy()
    labels_df = labels_df[labels_df["peerid"].isin(common)].copy()
    return edges, labels_df

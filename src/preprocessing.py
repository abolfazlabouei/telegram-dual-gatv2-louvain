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

import io

import array

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _peek_first_line(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.readline()


def _is_whole_row_quoted(first_line: str) -> bool:
    """True if the header row looks like `"col1,col2"` (one big quoted
    field) rather than `col1,col2` -- checked from a single line, so
    detection itself never touches the rest of a possibly huge file."""
    s = first_line.strip()
    return len(s) >= 2 and s[0] == '"' and s[-1] == '"' and "," in s[1:-1]


def _read_quoted_int_pair_csv(path: str, col_a: str, col_b: str) -> tuple[np.ndarray, np.ndarray]:
    """Stream-parse a CSV where every row is wrapped in one pair of quotes
    (`"userID,groupID"` instead of `userID,groupID`) and both requested
    columns are plain integers. Single pass, one line in memory at a time
    -- avoids pandas.read_csv entirely for this malformed case, since that
    would otherwise mean reading the whole file twice (once to discover
    the problem, once to reparse it) and is what caused the OOM kill on
    large membership files.

    Accumulates into `array.array('q')` rather than plain Python lists:
    a Python list of ints stores one boxed PyLong object per element
    (~28+ bytes each) plus an 8-byte pointer in the list, whereas
    `array.array` packs int64s contiguously like a C array (8 bytes
    each) -- roughly a 4-5x reduction in peak memory during accumulation
    for tens of millions of rows.
    """
    vals_a = array.array("q")
    vals_b = array.array("q")
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().strip('"')
        header_cols = [c.strip() for c in header.split(",")]
        try:
            idx_a = header_cols.index(col_a)
            idx_b = header_cols.index(col_b)
        except ValueError as e:
            raise ValueError(
                f"{path}: missing column(s). Found columns: {header_cols}. "
                f"Pass the matching --*-col flag to point at your real column name(s)."
            ) from e

        skipped = 0
        for line in f:
            line = line.strip().strip('"')
            if not line:
                continue
            parts = line.split(",")
            try:
                vals_a.append(int(parts[idx_a]))
                vals_b.append(int(parts[idx_b]))
            except (ValueError, IndexError):
                skipped += 1

    if skipped:
        print(f"[preprocessing] {path}: skipped {skipped:,} malformed row(s) while streaming")
    return np.frombuffer(vals_a, dtype=np.int64), np.frombuffer(vals_b, dtype=np.int64)


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
    on either side and exact duplicate rows.

    Memory-conscious by construction: only the two needed columns are ever
    materialized (via `usecols` + explicit `dtype` on the normal path, so
    pandas never allocates the other columns or a slower generic dtype),
    and the malformed "whole row wrapped in quotes" case is streamed
    line-by-line rather than parsed twice.
    """
    if _is_whole_row_quoted(_peek_first_line(path)):
        print(f"[preprocessing] {path}: detected fully-quoted rows; "
              f"using the memory-light streaming parser")
        user_ids, group_ids = _read_quoted_int_pair_csv(path, user_col, group_col)
        df = pd.DataFrame({"user_id": user_ids, "group_id": group_ids})
    else:
        df = pd.read_csv(path, usecols=[user_col, group_col],
                          dtype={user_col: np.int64, group_col: np.int64})
        df = df.rename(columns={user_col: "user_id", group_col: "group_id"})

    before = len(df)
    df = df.dropna().drop_duplicates()
    print(f"[preprocessing] memberships: {before:,} raw rows -> {len(df):,} after dropna + de-dup")
    return df


def _read_csv_maybe_quoted(path: str) -> pd.DataFrame:
    """Like `pd.read_csv(path)`, but also handles a file where every row is
    wrapped in one pair of quotes (see `_is_whole_row_quoted`). Fine to use
    here since the group metadata file is one row per *group* -- far
    smaller than the membership file, so reading it twice in the fallback
    case isn't a memory concern the way it was for memberships (see the
    streaming parser above, used there instead for exactly that reason).

    Note: the fallback here does a naive per-line comma split after
    stripping the outer quote, so if your `about` text itself contains a
    literal comma *and* your file also has this whole-row-quoting problem,
    a row could still split incorrectly. This hasn't come up in practice
    (the normal, non-quoted path handles embedded commas in text fields
    correctly via pandas' own CSV quoting) -- flagging it here in case it
    ever does.
    """
    if _is_whole_row_quoted(_peek_first_line(path)):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        cleaned_lines = []
        for line in lines:
            s = line.strip()
            if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
                s = s[1:-1]
            cleaned_lines.append(s)
        return pd.read_csv(io.StringIO("\n".join(cleaned_lines)))
    return pd.read_csv(path)


def load_group_metadata(path: str, group_col: str = "peerid", about_col: str = "about",
                         label_col: str = "label", name_col: str = "peer_name") -> pd.DataFrame:
    """One row per group, standardized to columns peerid/peer_name/about/label.
    If `name_col` isn't present, the group id is used as a placeholder name
    (so downstream text embedding still has *something* to encode)."""
    df = _read_csv_maybe_quoted(path)
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


def report_user_degree_distribution(memberships: pd.DataFrame) -> pd.Series:
    """Per-user group-count distribution -- printed unconditionally since
    it's directly relevant for the thesis's data-preprocessing writeup
    (e.g. "N outlier accounts joined in more than X groups and were
    excluded"), and used by `filter_hub_users` to pick a cutoff."""
    counts = memberships.groupby("user_id").size()
    print(f"[preprocessing] per-user group-count: mean={counts.mean():.2f}, "
          f"median={counts.median():.0f}, max={counts.max():,}, "
          f"p99={counts.quantile(0.99):.0f}, p99.9={counts.quantile(0.999):.0f}")
    return counts


def filter_hub_users(memberships: pd.DataFrame, max_groups_per_user: int | None = None,
                      hub_percentile: float = 99.5) -> pd.DataFrame:
    """Drop membership rows for outlier "hub" users who belong to an
    unusually large number of groups (bots, spam/aggregator accounts, or
    admin accounts subscribed to huge numbers of channels).

    These users are why the bipartite-projection matrix multiplication in
    `compute_shared_membership_edges` (M^T @ M) can blow up in memory: a
    single user in k groups contributes up to k^2 entries to the output
    co-membership matrix, so a handful of users with k in the thousands
    can dominate both runtime and memory on their own.

    If `max_groups_per_user` is not given, the cutoff defaults to the
    `hub_percentile`-th percentile of the per-user group-count
    distribution, so it adapts to your actual data instead of a fixed
    guess. Pass `max_groups_per_user=None` together with a very high
    `hub_percentile` (e.g. 100) -- or skip calling this function entirely
    -- to disable filtering.
    """
    counts = report_user_degree_distribution(memberships)

    if max_groups_per_user is None:
        max_groups_per_user = max(int(counts.quantile(hub_percentile / 100)), 1)
    hub_users = counts[counts > max_groups_per_user].index

    if len(hub_users) == 0:
        print(f"[preprocessing] no users exceed {max_groups_per_user:,} groups; nothing filtered")
        return memberships

    before_rows = len(memberships)
    memberships = memberships[~memberships["user_id"].isin(hub_users)].copy()
    print(f"[preprocessing] removed {len(hub_users):,} hub user(s) (> {max_groups_per_user:,} "
          f"groups each): {before_rows:,} -> {len(memberships):,} membership rows "
          f"({before_rows - len(memberships):,} rows dropped)")
    return memberships


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

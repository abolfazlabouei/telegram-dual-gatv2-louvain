#!/usr/bin/env python
"""
Build `data/telegram_graph.edgelist` and `data/graph_labels.csv` (the
files `scripts/run_pipeline.py` expects) from two raw files:

  1. a membership table: one row per (user, group) pair
  2. a group metadata table: one row per group (id, description, label[, name])

Usage (defaults match the column names used elsewhere in this repo):

    python -m scripts.prepare_data \\
        --memberships raw/memberships.csv \\
        --groups raw/groups.csv \\
        --output-dir data \\
        --min-shared 5

If your raw files use different column names, point at them explicitly:

    python -m scripts.prepare_data \\
        --memberships raw/memberships.csv --user-col uid --group-col-membership chat_id \\
        --groups raw/groups.csv --group-col-meta chat_id --about-col description --label-col category \\
        --output-dir data

Run `python -m scripts.prepare_data --help` for the full flag list.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import (
    load_membership_table,
    load_group_metadata,
    compute_shared_membership_edges,
    restrict_to_common_nodes,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--memberships", required=True, help="raw (user, group) membership CSV")
    p.add_argument("--groups", required=True,
                    help="raw group metadata CSV (id, about/description, label[, name])")
    p.add_argument("--output-dir", default="data")
    p.add_argument("--min-shared", type=int, default=5,
                    help="minimum shared members for two groups to get an edge (default: 5, "
                         "matches the thesis's structural-graph threshold)")

    p.add_argument("--user-col", default="user_id",
                    help="column in --memberships holding the numeric user id")
    p.add_argument("--group-col-membership", default="group_id",
                    help="column in --memberships holding the numeric group id")
    p.add_argument("--group-col-meta", default="peerid",
                    help="column in --groups holding the numeric group id")
    p.add_argument("--about-col", default="about",
                    help="column in --groups holding the group's text description")
    p.add_argument("--label-col", default="label",
                    help="column in --groups holding the reference/ground-truth label")
    p.add_argument("--name-col", default="peer_name",
                    help="column in --groups holding a display name (optional; "
                         "falls back to the group id if not found)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== 1) Loading raw files ===")
    memberships = load_membership_table(
        args.memberships, user_col=args.user_col, group_col=args.group_col_membership,
    )
    labels_df = load_group_metadata(
        args.groups, group_col=args.group_col_meta, about_col=args.about_col,
        label_col=args.label_col, name_col=args.name_col,
    )

    print("\n=== 2) Computing shared-membership edges ===")
    edges = compute_shared_membership_edges(memberships, min_shared=args.min_shared)

    print("\n=== 3) Restricting to nodes common to both files ===")
    edges, labels_df = restrict_to_common_nodes(edges, labels_df)

    if len(edges) == 0:
        print("\n[WARNING] No edges survived filtering. Likely causes: group ids don't "
              "actually match between the two files (check --group-col-membership vs "
              "--group-col-meta refer to the *same* id space), or --min-shared is too high.")

    edges_path = os.path.join(args.output_dir, "telegram_graph.edgelist")
    labels_path = os.path.join(args.output_dir, "graph_labels.csv")
    edges[["src", "dst", "weight"]].to_csv(edges_path, sep=" ", header=False, index=False)
    labels_df.to_csv(labels_path, index=False)

    print(f"\nWrote {len(edges):,} edges     -> {edges_path}")
    print(f"Wrote {len(labels_df):,} groups -> {labels_path}")
    print(f"\nNext: python -m scripts.run_pipeline --edges {edges_path} --labels {labels_path}")


if __name__ == "__main__":
    main()

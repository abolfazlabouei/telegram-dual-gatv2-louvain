"""
One-stop "make me every figure and its underlying numbers" pass over a
graph, built specifically so you can drop the PNGs straight into slides
and the CSVs straight into a thesis table -- without hunting through
plotting code to find which array fed which chart.

Every plot is saved twice:
  - as a PNG under `fig_dir` (300 dpi, print-quality), and
  - as the exact numbers behind it under `data_dir`, as a tidy
    one- or two-column CSV with a header row.

Call `graph_report(...)` once per graph (structural, content, fused --
whichever you have) with a distinct `name`; everything gets that name
as a filename prefix so outputs from different graphs never collide.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import networkx as nx

from . import graph_stats as gs
from . import visualize as viz


def _save_values_csv(path: str, column_name: str, values) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame({column_name: values}).to_csv(path, index=False)


def _save_xy_csv(path: str, x_name: str, x, y_name: str, y) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame({x_name: x, y_name: y}).to_csv(path, index=False)


def graph_report(
    G: nx.Graph,
    name: str,
    fig_dir: str,
    data_dir: str,
    weighted: bool = True,
    run_power_law: bool = False,
    save_plots: bool = True,
) -> dict:
    """Compute every stat + plot this repo knows how to make for one
    graph, save everything under `fig_dir`/`data_dir` prefixed with
    `name`, and return the one-row summary dict (also written to
    `<name>_summary.csv` -- ready to paste straight into a thesis table
    like the "graph statistics" tables in Chapter 5).

    Produces (for `name="structural"`, as an example):
      figures/structural_degree_dist.png       + data_for_figures/structural_degree_values.csv
      figures/structural_degree_ccdf.png       + data_for_figures/structural_degree_ccdf.csv
      figures/structural_rank_size.png         (reuses the degree values csv)
      figures/structural_weight_dist.png       + data_for_figures/structural_weight_values.csv
      figures/structural_clustering_vs_degree.png + data_for_figures/structural_clustering_vs_degree.csv
      figures/structural_k_core_curve.png      + data_for_figures/structural_k_core_curve.csv
      figures/structural_component_sizes.png   + data_for_figures/structural_component_sizes.csv
      data_for_figures/structural_summary.csv  (nodes, edges, density, avg_degree, avg_clustering, ...)
      (+ structural_power_law_fit.png/.json if run_power_law=True and the
        optional `powerlaw` package is installed)
    """
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    degrees = np.array([d for _, d in G.degree()])
    stats = gs.basic_network_stats(G)

    # ---- degree distribution ----
    _save_values_csv(os.path.join(data_dir, f"{name}_degree_values.csv"), "degree", degrees)
    if save_plots:
        viz.plot_degree_distribution(degrees, save_path=os.path.join(fig_dir, f"{name}_degree_dist.png"))

    # ---- degree CCDF (log-log tail) ----
    xs, ccdf = gs.degree_ccdf(degrees)
    if len(xs):
        _save_xy_csv(os.path.join(data_dir, f"{name}_degree_ccdf.csv"), "degree", xs, "ccdf", ccdf)
        if save_plots:
            viz.plot_degree_ccdf(xs, ccdf, save_path=os.path.join(fig_dir, f"{name}_degree_ccdf.png"))

    # ---- rank-size / Zipf (reuses the degree values, own plot) ----
    if save_plots and len(degrees):
        viz.plot_rank_size(degrees, save_path=os.path.join(fig_dir, f"{name}_rank_size.png"))

    # ---- edge weight distribution ----
    if weighted and G.number_of_edges():
        weights = np.array([d.get("weight", 1.0) for _, _, d in G.edges(data=True)])
        _save_values_csv(os.path.join(data_dir, f"{name}_weight_values.csv"), "weight", weights)
        if save_plots:
            viz.plot_weight_distribution(weights, save_path=os.path.join(fig_dir, f"{name}_weight_dist.png"))

    # ---- local clustering coefficient vs degree ----
    xs_c, ys_c = gs.clustering_vs_degree(G)
    if len(xs_c):
        _save_xy_csv(os.path.join(data_dir, f"{name}_clustering_vs_degree.csv"),
                     "degree", xs_c, "avg_clustering", ys_c)
        if save_plots:
            viz.plot_clustering_vs_degree(
                xs_c, ys_c, save_path=os.path.join(fig_dir, f"{name}_clustering_vs_degree.png"))

    # ---- k-core curve ----
    ks, sizes = gs.k_core_curve(G)
    if ks:
        _save_xy_csv(os.path.join(data_dir, f"{name}_k_core_curve.csv"), "k", ks, "num_nodes", sizes)
        if save_plots:
            viz.plot_k_core_curve(ks, sizes, save_path=os.path.join(fig_dir, f"{name}_k_core_curve.png"))

    # ---- connected components ----
    if G.number_of_edges():
        edges_idx = np.array(list(G.edges()), dtype=np.int64)
    else:
        edges_idx = np.zeros((0, 2), dtype=np.int64)
    cc = gs.connected_components_report(edges_idx, G.number_of_nodes())
    _save_values_csv(os.path.join(data_dir, f"{name}_component_sizes.csv"),
                      "component_size", cc["component_sizes"])
    if save_plots:
        viz.plot_component_sizes(cc["component_sizes"],
                                  save_path=os.path.join(fig_dir, f"{name}_component_sizes.png"))
    stats["num_components"] = cc["num_components"]
    stats["giant_component_fraction"] = cc["giant_component_fraction"]

    # ---- optional power-law fit on the degree tail ----
    if run_power_law:
        fit_result = gs.power_law_fit(degrees)
        if fit_result:
            fit_meta = {k: v for k, v in fit_result.items() if k != "fit_object"}
            with open(os.path.join(data_dir, f"{name}_power_law_fit.json"), "w") as f:
                json.dump(fit_meta, f, indent=2)
            if save_plots:
                viz.plot_power_law_fit(fit_result, save_path=os.path.join(fig_dir, f"{name}_power_law_fit.png"))
            stats.update({f"power_law_{k}": v for k, v in fit_meta.items()})

    # ---- one-row summary table, ready to paste into a thesis table ----
    pd.DataFrame([{"graph": name, **stats}]).to_csv(
        os.path.join(data_dir, f"{name}_summary.csv"), index=False)

    return stats


def save_community_report(
    pred: np.ndarray,
    fig_dir: str,
    data_dir: str,
    name: str = "fused",
    save_plots: bool = True,
) -> pd.DataFrame:
    """Save the community-size distribution as both a PNG and a tidy
    `(community_id, size)` CSV, sorted largest-first -- handy for a
    "top N communities" table."""
    os.makedirs(data_dir, exist_ok=True)
    sizes = pd.Series(pred).value_counts().rename_axis("community_id").reset_index(name="size")
    sizes = sizes.sort_values("size", ascending=False).reset_index(drop=True)
    sizes.to_csv(os.path.join(data_dir, f"{name}_community_sizes.csv"), index=False)

    if save_plots:
        viz.plot_community_sizes(
            pred, title=f"Community size distribution ({name})",
            save_path=os.path.join(fig_dir, f"{name}_community_sizes.png"))

    return sizes


def save_alpha_report(
    alpha: np.ndarray,
    idx2id: dict,
    fig_dir: str,
    data_dir: str,
    save_plots: bool = True,
) -> None:
    """Save the fusion-gate alpha values as both a PNG histogram and a
    per-node CSV (`peerid, alpha`)."""
    os.makedirs(data_dir, exist_ok=True)
    pd.DataFrame({
        "peerid": [idx2id[i] for i in range(len(alpha))],
        "alpha": alpha,
    }).to_csv(os.path.join(data_dir, "gate_alpha.csv"), index=False)

    if save_plots:
        viz.plot_alpha_histogram(alpha, save_path=os.path.join(fig_dir, "gate_alpha_hist.png"))

"""
Community-detection evaluation metrics.

`label` in `graph_labels.csv` is treated as the reference/ground-truth
grouping for NMI, ARI and Purity. Document where that column actually
comes from for your dataset -- if it was itself derived from group
name/description text, evaluating the *content* graph against it risks
data leakage (the same signal appears on both sides of the comparison).
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from community import modularity as louvain_modularity
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    C = contingency_matrix(y_true, y_pred)
    return float(np.sum(np.amax(C, axis=0)) / np.sum(C))


def eval_partition(G: nx.Graph, partition: dict, y_true: np.ndarray):
    """Compute Modularity/NMI/ARI/Purity and the number of communities for
    one partition. Returns (Q, NMI, ARI, Purity, num_communities, pred)."""
    pred = np.array([partition[i] for i in range(len(partition))], dtype=int)
    Q = louvain_modularity(partition, G, weight="weight")
    NMI = normalized_mutual_info_score(y_true, pred)
    ARI = adjusted_rand_score(y_true, pred)
    PUR = purity_score(y_true, pred)
    K = len(set(pred))
    return Q, NMI, ARI, PUR, K, pred


def collect_metrics_row(name: str, G: nx.Graph, partition: dict, y_true: np.ndarray,
                         extra_params: dict) -> dict:
    """One tidy dict/row per run, suitable for `pd.DataFrame([...])` and
    appending to a running results CSV."""
    Q, NMI, ARI, PUR, K, _ = eval_partition(G, partition, y_true)
    return {
        "run_name": name, "Q": Q, "#C": K, "NMI": NMI, "ARI": ARI, "Purity": PUR,
        **extra_params,
    }


def build_label_index(labels_df, column: str = "label") -> tuple[np.ndarray, dict]:
    """Map the raw reference-label column to contiguous integers, matching
    the node ordering used everywhere else (i.e. `labels_df` row order).

    Missing labels (empty cells) are pandas NaN -- a float, not a string --
    even after `.astype(str)` (pandas leaves NaN as-is rather than turning
    it into the text "nan"). Left alone, that mixes floats and strings in
    the same column and `sorted()` crashes comparing them. Filled here
    with an explicit "missing_label" category instead, with a count
    printed so you know if this is happening on a meaningful number of
    rows (worth mentioning in the thesis if so).
    """
    n_missing = int(labels_df[column].isna().sum())
    if n_missing:
        print(f"[evaluate] {n_missing:,} row(s) have a missing '{column}' value; "
              f"grouping them into a single 'missing_label' category")

    y_raw = labels_df[column].fillna("missing_label").astype(str).tolist()
    label2int = {l: i for i, l in enumerate(sorted(set(y_raw)))}
    y_true = np.array([label2int[l] for l in y_raw], dtype=int)
    return y_true, label2int

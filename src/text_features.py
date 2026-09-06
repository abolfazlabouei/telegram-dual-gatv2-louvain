"""
Turn each group's (name, about) text into a numeric feature vector:
a multilingual sentence embedding plus a handful of cheap linguistic
features (name length, emoji count, presence of a link).

The sentence-embedding step is cached to disk (`emb_cache_path`) since it
is the slowest part of preprocessing and is deterministic given the model
and the input texts.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def build_group_texts(labels_df: pd.DataFrame) -> list[str]:
    """Concatenate name + about into one string per group, `[SEP]`-joined."""
    return (labels_df["peer_name"].fillna("") + " [SEP] " + labels_df["about"].fillna("")).tolist()


def compute_text_embeddings(
    labels_df: pd.DataFrame,
    model_name: str,
    cache_path: str,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """Encode every group's text with a SentenceTransformer model.

    Loads from `cache_path` if it already exists, otherwise computes and
    saves there. Delete the cache file to force recomputation (e.g. after
    changing `model_name`).
    """
    if os.path.exists(cache_path):
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer

    texts = build_group_texts(labels_df)
    model = SentenceTransformer(model_name, device=device)
    X_text = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.save(cache_path, X_text)
    return X_text


def _count_emojis(s: str) -> int:
    # crude heuristic: most emoji code points sit well above the Latin/CJK
    # ranges typically found in group names -- good enough as a cheap signal,
    # not a rigorous emoji detector.
    return sum(1 for ch in s if ord(ch) > 10000)


def _has_link(s: str) -> int:
    s = s.lower()
    return int("http://" in s or "https://" in s or "t.me/" in s)


def compute_linguistic_features(labels_df: pd.DataFrame) -> np.ndarray:
    """Standardized [name_length, emoji_count, has_link] per group."""
    name_lens = labels_df["peer_name"].fillna("").apply(len).to_numpy(dtype=np.float32)[:, None]
    emoji_cnt = labels_df["peer_name"].fillna("").apply(_count_emojis).to_numpy(dtype=np.float32)[:, None]
    link_flag = labels_df["about"].fillna("").apply(_has_link).to_numpy(dtype=np.float32)[:, None]

    ling = np.hstack([name_lens, emoji_cnt, link_flag]).astype(np.float32)
    ling = (ling - ling.mean(axis=0)) / (ling.std(axis=0) + 1e-8)
    return ling


def build_feature_matrix(
    X_text: np.ndarray,
    ling: np.ndarray,
    use_pca: bool = False,
    pca_dim: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """Concatenate sentence embeddings with linguistic features, with an
    optional PCA reduction of the sentence embeddings applied first."""
    if use_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_dim, random_state=seed)
        X_text = pca.fit_transform(X_text).astype(np.float32)

    return np.hstack([X_text.astype(np.float32), ling]).astype(np.float32)

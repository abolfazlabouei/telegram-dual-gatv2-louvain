# Telegram Dual-Stream GATv2 + Louvain

Community detection on Telegram groups by fusing two complementary graphs:

- a **structural** graph (edges from shared-membership counts between groups),
- a **content/text-similarity** graph (a mutual k-NN graph over sentence
  embeddings of each group's name + description).

A two-stream **GATv2** encoder learns a representation per graph, a small
learned gate fuses the two per node, and **Louvain** is run on a k-NN graph
reconstructed from the fused embeddings to produce the final communities.

This repo is a cleaned-up, modularized version of an exploratory Kaggle
notebook. It reproduces the same computations (same defaults, same
formulas), just split into importable modules instead of one long notebook,
with no Kaggle-specific paths or inline `!pip install` cells.

## Repository layout

```
src/
  preprocessing.py  # raw membership + group-metadata files -> edgelist + labels
  config.py         # all hyperparameters (dataclass) + CLI overrides
  data.py           # load edges/labels, sanity checks
  graph_stats.py    # connected components, degree/clustering stats, power-law fit
  text_features.py  # sentence embeddings + linguistic features (cached)
  graphs.py         # build structural graph & text-similarity kNN graph (PyG Data)
  models.py         # GATv2Encoder / FusionGate / DualGATv2
  losses.py         # reconstruction loss + cross-stream alignment loss
  train.py          # mini-batch training loop (NeighborLoader, AMP, early stopping)
  embeddings.py     # full-graph inference + stream fusion
  clustering.py     # fused-embedding kNN graph, Louvain, structural-only baseline
  evaluate.py       # NMI / ARI / Purity / Modularity
  visualize.py      # every plot from the notebook, save-to-file friendly
scripts/
  prepare_data.py   # raw files -> data/telegram_graph.edgelist + data/graph_labels.csv
  run_pipeline.py   # end-to-end: data -> training -> clustering -> evaluation
  run_baselines.py  # classical baselines (Louvain/Leiden/Infomap/Label Propagation) on the structural graph
  run_ablation.py   # retrains from scratch: no_align / struct_only / content_only ablations
  run_eval_only.py  # re-cluster/re-evaluate from a saved embedding file (no retraining)
```

## Step 0 — do you need `prepare_data.py`?

If you already have a ready-made `telegram_graph.edgelist` + `graph_labels.csv`
(see "Data format" below), skip straight to **Installation**.

If instead you have two *raw* files --- (1) a membership table of which
numeric user id belongs to which numeric group id, and (2) a group
metadata table with each group's numeric id, description, and reference
label --- run this first:

```bash
python -m scripts.prepare_data \
  --memberships raw/memberships.csv \
  --groups raw/groups.csv \
  --output-dir data \
  --min-shared 5
```

This computes, for every pair of groups, how many members they share
(via one sparse matrix multiplication, not a nested loop -- scales to
millions of membership rows), keeps only pairs with `weight >= 5`
(matching the thesis's structural-graph threshold), and restricts both
files down to the group ids common to both (a group with no metadata
can't be text-embedded; a group with no edges is an isolated node the
structural graph gets no signal from anyway).

Before computing edges, it also drops outlier "hub" users who belong to
an unusually large number of groups (bots, spam/aggregator accounts, or
admin accounts subscribed to thousands of channels) -- a handful of such
users can each single-handedly blow up the co-membership computation's
memory use (one user in k groups contributes up to k² entries). The
cutoff defaults to the 99.5th percentile of your data's own per-user
group-count distribution (printed either way); override with
`--max-groups-per-user N`, adjust with `--hub-percentile`, or disable
entirely with `--no-hub-filter`.

If your raw files use different column names than `user_id` / `group_id`
/ `peerid` / `about` / `label` / `peer_name`, point at the real ones:

```bash
python -m scripts.prepare_data \
  --memberships raw/memberships.csv --user-col uid --group-col-membership chat_id \
  --groups raw/groups.csv --group-col-meta chat_id --about-col description --label-col category \
  --output-dir data
```

Run `python -m scripts.prepare_data --help` for the full flag list. It
prints row counts at every step (raw rows loaded, edges above threshold,
groups common to both files) so you can sanity-check nothing silently
dropped to zero.

## Data format

Two *raw* input files (put your own copies somewhere, e.g. `raw/` --
not tracked in git):

- **membership table** -- one row per (user, group) pair, any CSV with
  a numeric user-id column and a numeric group-id column (names
  configurable, see Step 0 above).
- **group metadata table** -- one row per group, with a numeric id, a
  text description, and a reference label (names configurable too).

`scripts/prepare_data.py` turns these into the two files the pipeline
actually reads, under `data/` (gitignored -- put your own copies there):

- **`telegram_graph.edgelist`** -- whitespace-separated, no header:
  ```
  src dst weight
  ```
  `src`/`dst` are the same group ids used in the labels file; `weight`
  is the number of members shared between the two groups (thresholded
  at `--min-shared`, default 5).

- **`graph_labels.csv`** -- columns `peerid, peer_name, about, label`.
  `label` is whatever reference grouping you evaluate NMI/ARI/Purity
  against.

  > **Document this column before you publish.** If it was derived from
  > `peer_name`/`about` (e.g. a topic tag mined from the same text used
  > to build the content graph), evaluating that graph against it is
  > circular. Write down the provenance here once you know it.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`torch-geometric`'s accelerator packages (`torch-scatter`, `torch-sparse`,
`pyg-lib`) ship as wheels matched to your exact torch + CUDA build and are
**not** on plain PyPI for every combination. After installing `torch`,
check its CUDA version and then run, e.g.:

```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.6.0+cu124.html   # optional, faster
```

(swap `torch-2.6.0+cu124` for whatever `python -c "import torch; print(torch.__version__, torch.version.cuda)"` reports).

## Running

```bash
python -m scripts.run_pipeline \
  --edges data/telegram_graph.edgelist \
  --labels data/graph_labels.csv \
  --output-dir outputs
```

Useful overrides (see `src/config.py` / `--help` for the full list):

```bash
python -m scripts.run_pipeline \
  --edges data/telegram_graph.edgelist --labels data/graph_labels.csv \
  --epochs 30 --batch-size 2048 --k-fused 12 --louvain-resolution 1.2 \
  --device cuda --output-dir outputs/run2
```

Everything not exposed as a flag (loss weights, GATv2 hidden sizes,
NeighborLoader fan-out, etc.) can be changed either by editing the
defaults in `src/config.py`, or programmatically:

```python
from src.config import Config
from scripts.run_pipeline import main
main(Config(lambda_a=0.0, hidden_dim=96))
```

### Outputs

Written to `--output-dir` (default `outputs/`):

- `fused_embeddings.npy`, `gate_alpha.npy` — final per-node embeddings and gate weights (raw arrays, for reuse in code)
- `communities_fused.csv` — `peerid, community` assignment from the proposed method
- `training_loss.csv` — per-epoch training loss
- `k_sensitivity.csv` — NMI/ARI/Purity/Modularity vs. k for the fused kNN graph (if enabled)
- `results_summary.csv` — one row per run (proposed + structural-only baseline)
- `config_used.json` — the exact config the run used, for reproducibility

**`figures/`** — every plot as a 300-dpi PNG, ready to paste into slides or
the thesis: degree/weight distributions, degree CCDF, rank-size (Zipf),
local-clustering-vs-degree, k-core curve, connected-component sizes, and
one-row summary stats -- generated for the **structural**, **content**
(text-similarity), and **fused** graphs separately (filenames prefixed
`structural_`, `content_`, `fused_`), plus the fusion-gate alpha histogram
and the final community-size distribution.

**`data_for_figures/`** — the exact numbers behind every one of those PNGs,
as tidy CSVs (one row per data point, e.g. `structural_degree_values.csv`
has one `degree` column), so you can redraw any chart yourself in Excel,
matplotlib, or whatever your thesis template needs, without re-running the
pipeline. Also includes:
- `<graph>_summary.csv` per graph (nodes, edges, density, avg degree,
  avg clustering, assortativity, #components, giant-component fraction)
  -- these map directly onto the "graph statistics" tables in Chapter 5
- `all_graph_summaries.csv` -- the three `_summary.csv` files stacked into one table
- `gate_alpha.csv` -- per-node `(peerid, alpha)`
- `fused_community_sizes.csv` -- `(community_id, size)`, largest first

Pass `--power-law-fit` to also fit and plot a power-law curve on the
structural graph's degree tail (needs `pip install powerlaw`; skipped
with a printed note if not installed).

### Re-clustering without retraining

If you already have `fused_embeddings.npy` and just want to sweep
`k_fused` / `louvain_resolution`:

```bash
python -m scripts.run_eval_only \
  --labels data/graph_labels.csv \
  --embeddings outputs/fused_embeddings.npy \
  --k-fused 12 --louvain-resolution 1.2 \
  --output-dir outputs/resweep_k12
```

### Classical baselines (Louvain / Leiden / Infomap / Label Propagation)

For the Chapter 5 "comparison with classical methods" table, run:

```bash
python -m scripts.run_baselines \
  --edges data/telegram_graph.edgelist \
  --labels data/graph_labels.csv \
  --output-dir outputs
```

This only needs the structural graph -- no GATv2 training, no text
embeddings -- so it finishes in seconds to minutes. It runs each
algorithm in both a weighted and unweighted variant and writes
`outputs/classical_baselines.csv` with one row per (algorithm, weighted)
combination: `Modularity, NMI, ARI, Purity, num_communities, runtime_s`.

Louvain and Label Propagation need nothing beyond `requirements.txt`.
Leiden needs `pip install python-igraph leidenalg`; Infomap needs
`pip install infomap`. Either missing dependency is skipped with a
printed note (not a crash) -- rerun after installing to fill in the rest
of the table.

### Ablation studies

Three retrain-from-scratch ablations, using the exact same data and
hyperparameters as the main run so they're directly comparable in a
thesis table:

```bash
python -m scripts.run_ablation --mode all \
  --edges data/telegram_graph.edgelist --labels data/graph_labels.csv \
  --output-dir outputs/ablation
```

- `no_align` -- full dual-stream model, but with the alignment loss
  (`lambda_a`) switched off, isolating whether it actually helps
- `struct_only` -- a single GATv2 encoder trained only on the structural
  graph, clustered directly on its output (no content graph at all)
- `content_only` -- the same, but only on the content graph

Run one mode at a time (`--mode no_align`, `--mode struct_only`, or
`--mode content_only`) instead of `all` if you'd rather not run all
three back to back -- each call appends to (and replaces same-named rows
in) `outputs/ablation/ablation_results.csv`, so partial runs across
multiple sessions still end up as one combined table. Its columns match
`results_summary.csv` (`run_name, Q, #C, NMI, ARI, Purity, ...`), so the
two files concatenate directly into one comparison table.

## What this refactor does and does not fix

This is a structural cleanup, not a methodology fix. Kept intentionally
as-is:

- the training objective is exactly `lambda_s * recon(struct) + lambda_t *
  recon(text) + lambda_a * alignment_loss(struct, text)` — the alignment
  term pulls the two streams together, which works against keeping them
  complementary; see `src/losses.py` docstring;
- the fusion is a convex combination `alpha*h_struct + (1-alpha)*h_text`
  (`src/models.py::FusionGate`), not a concatenation-then-linear-layer —
  make sure any written description of the model matches this;
- classical-baseline comparison (Louvain, Leiden, Infomap, Label
  Propagation, weighted + unweighted) lives in `scripts/run_baselines.py`
  / `src/baselines.py` -- separate from the proposed method's own
  structural-only baseline in `results_summary.csv`, since it only needs
  the structural graph and runs in seconds.

## Reproducibility notes

- `src.train.set_seed` seeds `random`, `numpy`, and `torch` (CPU + all
  CUDA devices), but Louvain itself is stochastic in general — pass a
  fixed `--seed` and consider averaging metrics over a few seeds before
  reporting a single number.
- The SentenceTransformer embedding step is cached to
  `outputs/text_embeds.npy` (configurable via `--emb-cache-path` /
  `Config.emb_cache_path`); delete that file if you change
  `--sentence-model` or the input texts.

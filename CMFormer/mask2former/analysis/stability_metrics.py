"""
stability_metrics.py
Phase 1, Task 1.2 — Compute stability metrics on extracted attention data.

Metrics:
    1. JSD (Jensen-Shannon Divergence)
         - Per-layer, per-head, between two attention distributions
         - Cross-attention: only comparable within same scale group
             scale_groups = { 0: [0,3,6], 1: [1,4,7], 2: [2,5,8] }
         - Self-attention: all layers comparable (same [H, Q, Q] shape)

    2. Attention Entropy
         - H = -sum(p * log(p)) per head per layer
         - Measures how spread / focused each head's attention is
         - Computed independently per image (no pairing needed)

    3. Query Similarity (Cosine)
         - Pairwise cosine similarity of query embeddings across layers
         - Measures how much queries change as they flow through the decoder
         - Shape [Q, D] per layer — we compute mean cosine sim across all Q pairs

Usage (two modes):

    MODE A — single-domain stats (entropy, within-image layer progression):
        python scripts/run_stability_metrics.py \
            --input-dir experiments/stability_analysis/results/baseline_attention/cityscapes \
            --mode single \
            --output experiments/stability_analysis/results/cityscapes_metrics.csv

    MODE B — cross-domain JSD (source vs target):
        python scripts/run_stability_metrics.py \
            --source-dir experiments/stability_analysis/results/baseline_attention/cityscapes \
            --target-dir experiments/stability_analysis/results/baseline_attention/bdd100k \
            --mode cross \
            --output experiments/stability_analysis/results/cross_domain_jsd.csv

Place at:  ~/research/Mask2Former/mask2former/analysis/stability_metrics.py
"""

import os
import pickle
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — scale groups from the decoder's level_index = i % 3 cycling
# ---------------------------------------------------------------------------
# Layers that attend to the SAME pixel-feature scale (same K dimension).
# JSD is only valid between distributions of the same support size.
SCALE_GROUPS = {
    0: [0, 3, 6],   # K = 2048
    1: [1, 4, 7],   # K = 8192
    2: [2, 5, 8],   # K = 32768
}

# Inverse map: layer_idx -> scale_group_id
LAYER_TO_SCALE = {}
for grp_id, layers in SCALE_GROUPS.items():
    for l in layers:
        LAYER_TO_SCALE[l] = grp_id

EPS = 1e-10   # numerical stability for log


# ---------------------------------------------------------------------------
# 0.  Interpolation helper for cross-domain JSD
# ---------------------------------------------------------------------------
def _interpolate_distribution(p: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample a 1-D probability distribution to a different length using
    linear interpolation, then renormalize so it sums to 1.
    This lets us compare attention distributions from images of different
    resolutions (different K dimensions).
    """
    if len(p) == target_len:
        return p
    x_old = np.linspace(0, 1, len(p))
    x_new = np.linspace(0, 1, target_len)
    p_interp = np.interp(x_new, x_old, p)
    p_interp = np.clip(p_interp, 0, None)          # no negatives from interp
    p_interp = p_interp / (p_interp.sum() + EPS)   # renormalize
    return p_interp


# ---------------------------------------------------------------------------
# 1.  Jensen-Shannon Divergence
# ---------------------------------------------------------------------------
def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) with eps-clipped q to avoid log(0)."""
    return np.sum(p * np.log(p / (q + EPS) + EPS))


def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon Divergence between two 1-D probability distributions.
    Both p and q must be non-negative and sum to ~1 (we renormalize just in case).
    Returns a value in [0, log(2)] ≈ [0, 0.693].

    For interpretability we return JSD / log(2) so the range is [0, 1].
    """
    p = p / (p.sum() + EPS)
    q = q / (q.sum() + EPS)
    m = 0.5 * (p + q)
    return 0.5 * (_kl(p, m) + _kl(q, m)) / np.log(2)   # normalized to [0,1]


def compute_jsd_per_head(attn_a: np.ndarray, attn_b: np.ndarray) -> np.ndarray:
    """
    Given two attention arrays of shape [H, K], compute JSD per head.
    Returns np.array of shape [H].
    """
    assert attn_a.shape == attn_b.shape, (
        f"Shape mismatch: {attn_a.shape} vs {attn_b.shape}. "
        "Are you comparing layers from the same scale group?"
    )
    H = attn_a.shape[0]
    jsds = np.zeros(H)
    for h in range(H):
        jsds[h] = compute_jsd(attn_a[h], attn_b[h])
    return jsds


# ---------------------------------------------------------------------------
# 2.  Attention Entropy
# ---------------------------------------------------------------------------
def compute_entropy(attn: np.ndarray) -> float:
    """
    Shannon entropy of a 1-D distribution.  attn should sum to ~1.
    H = -sum(p * log(p)).  Normalized by log(K) so range is [0, 1].
    """
    p = attn / (attn.sum() + EPS)
    K = len(p)
    return -np.sum(p * np.log(p + EPS)) / np.log(K)   # normalized


def compute_entropy_per_head(attn: np.ndarray) -> np.ndarray:
    """
    attn: [H, K]  (self_attn: [H, Q, Q] — caller should flatten Q×Q or pick a slice)
    Returns: [H] array of normalized entropies.
    """
    H = attn.shape[0]
    entropies = np.zeros(H)
    for h in range(H):
        entropies[h] = compute_entropy(attn[h].flatten())
    return entropies


# ---------------------------------------------------------------------------
# 3.  Query Cosine Similarity (layer-to-layer)
# ---------------------------------------------------------------------------
def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Per-query cosine similarity between two [Q, D] arrays.
    Returns [Q] array of cosine similarities.
    """
    # row-wise norms
    norm_a = np.linalg.norm(a, axis=1, keepdims=True) + EPS
    norm_b = np.linalg.norm(b, axis=1, keepdims=True) + EPS
    cos_sim = np.sum((a / norm_a) * (b / norm_b), axis=1)   # [Q]
    return cos_sim


def compute_query_similarity_consecutive(query_features: Dict[int, np.ndarray]) -> Dict[str, float]:
    """
    Compute mean cosine similarity between consecutive decoder layers' query features.
    e.g. layer0→layer1, layer1→layer2, ..., layer7→layer8.

    Returns dict:  { "q_sim_0_1": float, "q_sim_1_2": float, ... }
    """
    layers = sorted(query_features.keys())
    results = {}
    for i in range(len(layers) - 1):
        l_a, l_b = layers[i], layers[i + 1]
        cos_sims = compute_cosine_similarity(query_features[l_a], query_features[l_b])
        results[f"q_sim_{l_a}_{l_b}"] = float(cos_sims.mean())
    return results


# ---------------------------------------------------------------------------
# 4.  High-level: compute all metrics for a single .pkl result
# ---------------------------------------------------------------------------
def compute_all_metrics_single(result: Dict) -> Dict:
    """
    Given one loaded attention result dict, compute:
        - entropy per layer per head (self_attn and cross_attn)
        - query similarity (consecutive layers)

    Returns a flat dict suitable for one row of a DataFrame.
    """
    row = {"image_id": result["image_id"]}

    # --- entropy: self-attention -------------------------------------------
    for layer_idx, sa in result["self_attention"].items():
        # sa: [H, Q, Q] — flatten Q×Q to get distribution over Q*Q entries per head
        ent = compute_entropy_per_head(sa.reshape(sa.shape[0], -1))   # [H]
        row[f"sa_entropy_layer{layer_idx}_mean"] = float(ent.mean())
        for h in range(len(ent)):
            row[f"sa_entropy_layer{layer_idx}_head{h}"] = float(ent[h])

    # --- entropy: cross-attention ------------------------------------------
    for layer_idx, ca in result["cross_attention"].items():
        # ca: [H, K]
        ent = compute_entropy_per_head(ca)   # [H]
        row[f"ca_entropy_layer{layer_idx}_mean"] = float(ent.mean())
        for h in range(len(ent)):
            row[f"ca_entropy_layer{layer_idx}_head{h}"] = float(ent[h])

    # --- query similarity --------------------------------------------------
    q_sims = compute_query_similarity_consecutive(result["query_features"])
    row.update(q_sims)

    return row


# ---------------------------------------------------------------------------
# 5.  High-level: cross-domain JSD between two result dicts
# ---------------------------------------------------------------------------
def compute_cross_domain_jsd(result_source: Dict, result_target: Dict) -> Dict:
    """
    Compute per-layer, per-head JSD between source and target attention.

    Self-attention: all 9 layers compared directly.
    Cross-attention: only layers within the same scale group are compared.

    Returns a flat dict:
        { "sa_jsd_layer0_mean": float, "sa_jsd_layer0_head0": float, ...
          "ca_jsd_layer0_mean": float, ... }
    """
    row = {
        "source_image_id": result_source["image_id"],
        "target_image_id": result_target["image_id"],
    }

    # --- self-attention JSD ------------------------------------------------
    for layer_idx in sorted(result_source["self_attention"].keys()):
        sa_s = result_source["self_attention"][layer_idx].reshape(8, -1)   # [H, Q*Q]
        sa_t = result_target["self_attention"][layer_idx].reshape(8, -1)
        jsds = compute_jsd_per_head(sa_s, sa_t)   # [H]
        row[f"sa_jsd_layer{layer_idx}_mean"] = float(jsds.mean())
        for h in range(len(jsds)):
            row[f"sa_jsd_layer{layer_idx}_head{h}"] = float(jsds[h])

    # --- cross-attention JSD (same-scale only) -----------------------------
    for layer_idx in sorted(result_source["cross_attention"].keys()):
        ca_s = result_source["cross_attention"][layer_idx]   # [H, K]
        ca_t = result_target["cross_attention"][layer_idx]   # [H, K]

        # K may differ across domains (different image resolutions).
        # Interpolate both to the larger K so JSD is computed on the same support.
        K_s, K_t = ca_s.shape[1], ca_t.shape[1]
        if K_s != K_t:
            target_K = max(K_s, K_t)
            logger.debug(
                f"cross_attn layer {layer_idx}: K mismatch {K_s} vs {K_t} "
                f"— interpolating both to {target_K}"
            )
            H = ca_s.shape[0]
            ca_s_interp = np.stack([_interpolate_distribution(ca_s[h], target_K) for h in range(H)])
            ca_t_interp = np.stack([_interpolate_distribution(ca_t[h], target_K) for h in range(H)])
        else:
            ca_s_interp, ca_t_interp = ca_s, ca_t

        jsds = compute_jsd_per_head(ca_s_interp, ca_t_interp)
        row[f"ca_jsd_layer{layer_idx}_mean"] = float(jsds.mean())
        for h in range(len(jsds)):
            row[f"ca_jsd_layer{layer_idx}_head{h}"] = float(jsds[h])

    return row


# ---------------------------------------------------------------------------
# 6.  Batch helpers — load a directory of .pkl files
# ---------------------------------------------------------------------------
def load_all_results(input_dir: str) -> List[Dict]:
    """Load every attention_*.pkl in input_dir, sorted by image_id."""
    results = []
    for fname in sorted(os.listdir(input_dir)):
        if fname.startswith("attention_") and fname.endswith(".pkl"):
            with open(os.path.join(input_dir, fname), "rb") as f:
                results.append(pickle.load(f))
    logger.info(f"Loaded {len(results)} results from {input_dir}")
    return results


# ---------------------------------------------------------------------------
# 7.  Summary stats — aggregate per-layer means across all images
# ---------------------------------------------------------------------------
def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of per-image metrics, return a summary with
    mean and std of every numeric column.
    """
    return df.describe().loc[["mean", "std"]]
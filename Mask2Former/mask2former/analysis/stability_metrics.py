"""
stability_metrics.py
Compute Jensen-Shannon Divergence between attention distributions
extracted from two domains (source vs target).
 
Metrics:
    JSD per layer, per head, between source and target cross-attention
    and self-attention distributions.
 
    Cross-attention K may differ across domains (different image resolutions).
    We interpolate both to the same support before computing JSD.
 
Usage:
    from mask2former.analysis.stability_metrics import (
        compute_cross_domain_jsd,
        load_all_results,
        summarize_metrics,
    )
"""
 
import os
import pickle
import logging
from typing import Dict, List
 
import numpy as np
import pandas as pd
 
logger = logging.getLogger(__name__)
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Scale groups from the decoder's level_index = i % 3 cycling.
# Layers within the same group attend to the same feature map resolution (same K).
SCALE_GROUPS = {
    0: [0, 3, 6],   # K = 2048  (coarsest)
    1: [1, 4, 7],   # K = 8192  (mid)
    2: [2, 5, 8],   # K = 32768 (finest)
}
 
# Inverse map: layer_idx -> scale_group_id
LAYER_TO_SCALE = {l: grp for grp, layers in SCALE_GROUPS.items() for l in layers}
 
EPS = 1e-10
 
 
# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------
def _interpolate_distribution(p: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample a 1-D probability distribution to target_len via linear
    interpolation, then renormalize to sum to 1.
    Needed when source and target images have different resolutions (different K).
    """
    if len(p) == target_len:
        return p
    x_old = np.linspace(0, 1, len(p))
    x_new = np.linspace(0, 1, target_len)
    p_out = np.interp(x_new, x_old, p)
    p_out = np.clip(p_out, 0, None)
    p_out = p_out / (p_out.sum() + EPS)
    return p_out
 
 
# ---------------------------------------------------------------------------
# JSD
# ---------------------------------------------------------------------------
def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q), numerically stable."""
    return float(np.sum(p * np.log(p / (q + EPS) + EPS)))
 
 
def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon Divergence between two 1-D distributions.
    Renormalizes inputs, returns value in [0, 1] (divided by log 2).
    """
    p = p / (p.sum() + EPS)
    q = q / (q.sum() + EPS)
    m = 0.5 * (p + q)
    return 0.5 * (_kl(p, m) + _kl(q, m)) / np.log(2)
 
 
def compute_jsd_per_head(attn_a: np.ndarray, attn_b: np.ndarray) -> np.ndarray:
    """
    attn_a, attn_b: [H, K] — must have the same shape.
    Returns: [H] JSD values, one per attention head.
    """
    assert attn_a.shape == attn_b.shape, (
        f"Shape mismatch: {attn_a.shape} vs {attn_b.shape}"
    )
    return np.array([compute_jsd(attn_a[h], attn_b[h]) for h in range(attn_a.shape[0])])
 
 
# ---------------------------------------------------------------------------
# Cross-domain JSD — one source/target pair
# ---------------------------------------------------------------------------
def compute_cross_domain_jsd(result_source: Dict, result_target: Dict) -> Dict:
    """
    Compute per-layer, per-head JSD between source and target attention dicts.
 
    Self-attention  [H, Q, Q]: flattened to [H, Q*Q], all 9 layers compared.
    Cross-attention [H, K]:    K-mismatch handled via interpolation.
 
    Returns a flat dict suitable for one DataFrame row:
        { "sa_jsd_layer0_mean": float, "sa_jsd_layer0_head0": float, ...
          "ca_jsd_layer0_mean": float, "ca_jsd_layer0_head0": float, ... }
    """
    row = {
        "source_image_id": result_source["image_id"],
        "target_image_id": result_target["image_id"],
    }
 
    # Self-attention JSD
    for layer_idx in sorted(result_source["self_attention"].keys()):
        sa_s = result_source["self_attention"][layer_idx].reshape(8, -1)  # [H, Q*Q]
        sa_t = result_target["self_attention"][layer_idx].reshape(8, -1)
        jsds = compute_jsd_per_head(sa_s, sa_t)
        row[f"sa_jsd_layer{layer_idx}_mean"] = float(jsds.mean())
        for h, v in enumerate(jsds):
            row[f"sa_jsd_layer{layer_idx}_head{h}"] = float(v)
 
    # Cross-attention JSD
    for layer_idx in sorted(result_source["cross_attention"].keys()):
        ca_s = result_source["cross_attention"][layer_idx]  # [H, K]
        ca_t = result_target["cross_attention"][layer_idx]  # [H, K]
 
        K_s, K_t = ca_s.shape[1], ca_t.shape[1]
        if K_s != K_t:
            target_K = max(K_s, K_t)
            logger.debug(f"Layer {layer_idx}: K mismatch {K_s} vs {K_t}, interpolating to {target_K}")
            H = ca_s.shape[0]
            ca_s = np.stack([_interpolate_distribution(ca_s[h], target_K) for h in range(H)])
            ca_t = np.stack([_interpolate_distribution(ca_t[h], target_K) for h in range(H)])
 
        jsds = compute_jsd_per_head(ca_s, ca_t)
        row[f"ca_jsd_layer{layer_idx}_mean"] = float(jsds.mean())
        for h, v in enumerate(jsds):
            row[f"ca_jsd_layer{layer_idx}_head{h}"] = float(v)
 
    return row
 
 
# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_all_results(input_dir: str) -> List[Dict]:
    """Load every attention_*.pkl in input_dir, sorted by filename."""
    results = []
    for fname in sorted(os.listdir(input_dir)):
        if fname.startswith("attention_") and fname.endswith(".pkl"):
            with open(os.path.join(input_dir, fname), "rb") as f:
                results.append(pickle.load(f))
    logger.info(f"Loaded {len(results)} results from {input_dir}")
    return results
 
 
def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean and std of every numeric column in df."""
    return df.describe().loc[["mean", "std"]]
"""
visualize_attention.py
Phase 1, Task 1.4 (bonus) — Visualize cross-attention as heatmaps overlaid on images.

Creates:
    1. Single-image attention grids (9 layers × 8 heads)
    2. Side-by-side domain comparisons (CS vs BDD/Mapillary, same layer/head)
    3. Aggregate attention maps (mean over heads, per layer)

Place at:  ~/research/Mask2Former/mask2former/analysis/visualize_attention.py
Run via:   python scripts/visualize_attention.py

The cross-attention we saved is [H, K] where K is the number of pixel features.
These features come from a multi-scale feature pyramid with 3 levels:
    - Scale 0 (layers 0,3,6): K = H_0 * W_0  (coarsest)
    - Scale 1 (layers 1,4,7): K = H_1 * W_1  (mid)
    - Scale 2 (layers 2,5,8): K = H_2 * W_2  (finest)

For visualization we need to reshape [K] back to [H_i, W_i] and upsample to the
original image size.
"""

import os
import pickle
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

# Scale group structure
SCALE_GROUPS = {
    0: [0, 3, 6],   # K = 2048 = 32×64 for 1024×2048 images
    1: [1, 4, 7],   # K = 8192 = 64×128
    2: [2, 5, 8],   # K = 32768 = 128×256
}

# Typical spatial shapes for each scale (for Cityscapes 1024×2048 input)
# These may vary slightly depending on exact model config
SCALE_SHAPES = {
    0: (32, 64),     # HxW for scale 0 features
    1: (64, 128),    # scale 1
    2: (128, 256),   # scale 2
}


def infer_spatial_shape(K: int) -> Tuple[int, int]:
    """
    Given K (total number of features), infer the spatial H×W.
    We assume roughly square aspect ratio 1:2 (like Cityscapes).
    """
    # Try common shapes first
    if K == 2048:
        return (32, 64)
    elif K == 8192:
        return (64, 128)
    elif K == 32768:
        return (128, 256)
    # BDD100K has different resolution
    elif K == 920:      # 23×40
        return (23, 40)
    elif K == 3680:     # 46×80
        return (46, 80)
    elif K == 14720:    # 92×160
        return (92, 160)
    # Generic fallback: assume 1:2 aspect ratio
    else:
        h = int(np.sqrt(K / 2))
        w = K // h
        return (h, w)


def reshape_attention_to_2d(attn: np.ndarray, layer_idx: int) -> np.ndarray:
    """
    attn: [K] — per-head attention distribution over pixel features.
    Returns: [H, W] spatial attention map.
    """
    K = len(attn)
    h, w = infer_spatial_shape(K)
    
    # Reshape [K] -> [H, W]
    if h * w != K:
        logger.warning(f"Layer {layer_idx}: K={K} doesn't match {h}×{w}={h*w}. Truncating/padding.")
        # Truncate or pad to fit
        target = h * w
        if K > target:
            attn = attn[:target]
        else:
            attn = np.pad(attn, (0, target - K), constant_values=0)
    
    return attn.reshape(h, w)


def overlay_heatmap_on_image(
    img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    img: [H, W, 3] uint8 RGB
    heatmap: [H, W] float in [0, 1]
    Returns: [H, W, 3] uint8 RGB overlay
    """
    # Upsample heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 255]
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


# ---------------------------------------------------------------------------
# 1. Single-image attention grid (9 layers × 8 heads)
# ---------------------------------------------------------------------------
def visualize_single_image_grid(
    result: Dict,
    image_path: str,
    output_path: str,
    layer_range: Optional[List[int]] = None
):
    """
    Create a grid showing all layers and heads for one image.
    
    result: loaded .pkl from attention_extractor
    image_path: path to the original image
    output_path: where to save the grid
    layer_range: which layers to show (default: all 9)
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    
    if layer_range is None:
        layer_range = list(range(9))
    
    num_layers = len(layer_range)
    num_heads = 8
    
    fig = plt.figure(figsize=(num_heads * 3, num_layers * 3))
    gs = gridspec.GridSpec(num_layers, num_heads, hspace=0.3, wspace=0.1)
    
    for row_idx, layer_idx in enumerate(layer_range):
        ca = result["cross_attention"][layer_idx]   # [H, K]
        
        for head_idx in range(num_heads):
            attn_1d = ca[head_idx]  # [K]
            
            # Reshape to 2D
            attn_2d = reshape_attention_to_2d(attn_1d, layer_idx)
            
            # Normalize to [0, 1]
            attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-10)
            
            # Overlay on image
            overlay = overlay_heatmap_on_image(img, attn_2d, alpha=0.6)
            
            ax = fig.add_subplot(gs[row_idx, head_idx])
            ax.imshow(overlay)
            ax.axis("off")
            
            if row_idx == 0:
                ax.set_title(f"Head {head_idx}", fontsize=10)
            if head_idx == 0:
                ax.set_ylabel(f"Layer {layer_idx}\n(S{layer_idx % 3})", fontsize=10, rotation=0, ha="right", va="center")
    
    plt.suptitle(f"Cross-Attention Heatmap — Image {result['image_id']}", fontsize=16, y=0.995)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved grid → {output_path}")


# ---------------------------------------------------------------------------
# 2. Side-by-side domain comparison (same layer/head, two domains)
# ---------------------------------------------------------------------------
def visualize_side_by_side_comparison(
    result_source: Dict,
    result_target: Dict,
    image_path_source: str,
    image_path_target: str,
    output_path: str,
    layer_idx: int = 8,
    head_idx: int = 0,
    show_original: bool = False
):
    """
    Show the same layer/head from two domains side by side.
    Useful for seeing how attention shifts under domain change.
    
    If show_original=True, creates a 2x2 grid:
        Row 1: source original | target original
        Row 2: source overlay  | target overlay
    """
    img_src = np.array(Image.open(image_path_source).convert("RGB"))
    img_tgt = np.array(Image.open(image_path_target).convert("RGB"))
    
    # Get attention
    ca_src = result_source["cross_attention"][layer_idx][head_idx]  # [K]
    ca_tgt = result_target["cross_attention"][layer_idx][head_idx]  # [K]
    
    # Reshape to 2D
    attn_src_2d = reshape_attention_to_2d(ca_src, layer_idx)
    attn_tgt_2d = reshape_attention_to_2d(ca_tgt, layer_idx)
    
    # Normalize
    attn_src_2d = (attn_src_2d - attn_src_2d.min()) / (attn_src_2d.max() - attn_src_2d.min() + 1e-10)
    attn_tgt_2d = (attn_tgt_2d - attn_tgt_2d.min()) / (attn_tgt_2d.max() - attn_tgt_2d.min() + 1e-10)
    
    # Overlay
    overlay_src = overlay_heatmap_on_image(img_src, attn_src_2d, alpha=0.6)
    overlay_tgt = overlay_heatmap_on_image(img_tgt, attn_tgt_2d, alpha=0.6)
    
    # Plot
    if show_original:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Row 1: originals
        axes[0, 0].imshow(img_src)
        axes[0, 0].set_title(f"Source (Cityscapes) - Original", fontsize=12)
        axes[0, 0].axis("off")
        
        axes[0, 1].imshow(img_tgt)
        axes[0, 1].set_title(f"Target - Original", fontsize=12)
        axes[0, 1].axis("off")
        
        # Row 2: overlays
        axes[1, 0].imshow(overlay_src)
        axes[1, 0].set_title(f"Source - Layer {layer_idx}, Head {head_idx}", fontsize=12)
        axes[1, 0].axis("off")
        
        axes[1, 1].imshow(overlay_tgt)
        axes[1, 1].set_title(f"Target - Layer {layer_idx}, Head {head_idx}", fontsize=12)
        axes[1, 1].axis("off")
        
        plt.suptitle("Cross-Attention: Source vs Target Domain", fontsize=16, y=0.98)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        axes[0].imshow(overlay_src)
        axes[0].set_title(f"Source (Cityscapes)\nLayer {layer_idx}, Head {head_idx}", fontsize=12)
        axes[0].axis("off")
        
        axes[1].imshow(overlay_tgt)
        axes[1].set_title(f"Target\nLayer {layer_idx}, Head {head_idx}", fontsize=12)
        axes[1].axis("off")
        
        plt.suptitle("Cross-Attention: Source vs Target Domain", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comparison → {output_path}")


# ---------------------------------------------------------------------------
# 3. Aggregate attention map (mean over all heads, per layer)
# ---------------------------------------------------------------------------
def visualize_aggregate_per_layer(
    result: Dict,
    image_path: str,
    output_path: str
):
    """
    Show 9 attention maps (one per layer), each averaged over all 8 heads.
    This gives a cleaner view of "where is the model looking overall".
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for layer_idx in range(9):
        ca = result["cross_attention"][layer_idx]   # [H, K]
        
        # Mean over heads
        attn_mean = ca.mean(axis=0)  # [K]
        
        # Reshape to 2D
        attn_2d = reshape_attention_to_2d(attn_mean, layer_idx)
        
        # Normalize
        attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-10)
        
        # Overlay
        overlay = overlay_heatmap_on_image(img, attn_2d, alpha=0.6)
        
        axes[layer_idx].imshow(overlay)
        axes[layer_idx].set_title(f"Layer {layer_idx} (Scale S{layer_idx % 3})", fontsize=12)
        axes[layer_idx].axis("off")
    
    plt.suptitle(f"Aggregate Cross-Attention (Mean Over Heads) — Image {result['image_id']}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved aggregate → {output_path}")


# ---------------------------------------------------------------------------
# 4. I/O helpers
# ---------------------------------------------------------------------------
def load_result(pkl_path: str) -> Dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)
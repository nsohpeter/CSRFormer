"""
visualize_predictions.py
Visualize predicted segmentation masks vs ground truth, alongside attention heatmaps.
"""

import os
import logging
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch

logger = logging.getLogger(__name__)


# Cityscapes official color palette (19 training classes)
CITYSCAPES_COLORS = np.array([
    [128, 64, 128],   # 0: road
    [244, 35, 232],   # 1: sidewalk
    [70, 70, 70],     # 2: building
    [102, 102, 156],  # 3: wall
    [190, 153, 153],  # 4: fence
    [153, 153, 153],  # 5: pole
    [250, 170, 30],   # 6: traffic light
    [220, 220, 0],    # 7: traffic sign
    [107, 142, 35],   # 8: vegetation
    [152, 251, 152],  # 9: terrain
    [70, 130, 180],   # 10: sky
    [220, 20, 60],    # 11: person
    [255, 0, 0],      # 12: rider
    [0, 0, 142],      # 13: car
    [0, 0, 70],       # 14: truck
    [0, 60, 100],     # 15: bus
    [0, 80, 100],     # 16: train
    [0, 0, 230],      # 17: motorcycle
    [119, 11, 32],    # 18: bicycle
], dtype=np.uint8)


def colorize_mask(mask: np.ndarray, dataset_name: str = "cityscapes") -> np.ndarray:
    """Convert semantic segmentation mask to RGB."""
    H, W = mask.shape
    colored = np.zeros((H, W, 3), dtype=np.uint8)
    
    unique_classes = np.unique(mask)
    logger.info(f"Colorizing mask - unique values: {unique_classes}")
    
    for cls_id in unique_classes:
        if cls_id == 255:
            colored[mask == cls_id] = [0, 0, 0]
        elif 0 <= cls_id < len(CITYSCAPES_COLORS):
            colored[mask == cls_id] = CITYSCAPES_COLORS[cls_id]
        else:
            colored[mask == cls_id] = [255, 0, 255]
            logger.warning(f"Invalid class ID: {cls_id}")
    
    return colored


def visualize_side_by_side_full(
    image_path_source: str,
    image_path_target: str,
    gt_mask_source: np.ndarray,
    gt_mask_target: np.ndarray,
    pred_mask_source: np.ndarray,
    pred_mask_target: np.ndarray,
    attention_result_source: Dict,
    attention_result_target: Dict,
    output_path: str,
    layer_idx: int = 8,
    head_idx: int = 0
):
    """Full side-by-side comparison."""
    from mask2former.analysis.visualize_attention import (
        reshape_attention_to_2d,
        overlay_heatmap_on_image
    )
    
    img_src = np.array(Image.open(image_path_source).convert("RGB"))
    img_tgt = np.array(Image.open(image_path_target).convert("RGB"))
    
    ca_src = attention_result_source["cross_attention"][layer_idx][head_idx]
    ca_tgt = attention_result_target["cross_attention"][layer_idx][head_idx]
    
    attn_src_2d = reshape_attention_to_2d(ca_src, layer_idx)
    attn_tgt_2d = reshape_attention_to_2d(ca_tgt, layer_idx)
    
    attn_src_2d = (attn_src_2d - attn_src_2d.min()) / (attn_src_2d.max() - attn_src_2d.min() + 1e-10)
    attn_tgt_2d = (attn_tgt_2d - attn_tgt_2d.min()) / (attn_tgt_2d.max() - attn_tgt_2d.min() + 1e-10)
    
    attn_overlay_src = overlay_heatmap_on_image(img_src, attn_src_2d, alpha=0.6)
    attn_overlay_tgt = overlay_heatmap_on_image(img_tgt, attn_tgt_2d, alpha=0.6)
    
    if gt_mask_source.shape != img_src.shape[:2]:
        gt_mask_source = cv2.resize(gt_mask_source.astype(np.uint8), (img_src.shape[1], img_src.shape[0]), interpolation=cv2.INTER_NEAREST)
    if gt_mask_target.shape != img_tgt.shape[:2]:
        gt_mask_target = cv2.resize(gt_mask_target.astype(np.uint8), (img_tgt.shape[1], img_tgt.shape[0]), interpolation=cv2.INTER_NEAREST)
    if pred_mask_source.shape != img_src.shape[:2]:
        pred_mask_source = cv2.resize(pred_mask_source.astype(np.uint8), (img_src.shape[1], img_src.shape[0]), interpolation=cv2.INTER_NEAREST)
    if pred_mask_target.shape != img_tgt.shape[:2]:
        pred_mask_target = cv2.resize(pred_mask_target.astype(np.uint8), (img_tgt.shape[1], img_tgt.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    gt_src_colored = colorize_mask(gt_mask_source, "cityscapes")
    gt_tgt_colored = colorize_mask(gt_mask_target, "bdd100k")
    pred_src_colored = colorize_mask(pred_mask_source, "cityscapes")
    pred_tgt_colored = colorize_mask(pred_mask_target, "bdd100k")
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    
    axes[0, 0].imshow(img_src)
    axes[0, 0].set_title("Source - Original", fontsize=12)
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(img_tgt)
    axes[0, 1].set_title("Target - Original", fontsize=12)
    axes[0, 1].axis("off")
    
    axes[1, 0].imshow(gt_src_colored)
    axes[1, 0].set_title("Source - Ground Truth", fontsize=12)
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(gt_tgt_colored)
    axes[1, 1].set_title("Target - Ground Truth", fontsize=12)
    axes[1, 1].axis("off")
    
    axes[2, 0].imshow(pred_src_colored)
    axes[2, 0].set_title("Source - Predicted", fontsize=12)
    axes[2, 0].axis("off")
    
    axes[2, 1].imshow(pred_tgt_colored)
    axes[2, 1].set_title("Target - Predicted", fontsize=12)
    axes[2, 1].axis("off")
    
    axes[3, 0].imshow(attn_overlay_src)
    axes[3, 0].set_title(f"Source - Attention (L{layer_idx}, H{head_idx})", fontsize=12)
    axes[3, 0].axis("off")
    
    axes[3, 1].imshow(attn_overlay_tgt)
    axes[3, 1].set_title(f"Target - Attention (L{layer_idx}, H{head_idx})", fontsize=12)
    axes[3, 1].axis("off")
    
    plt.suptitle("Full Comparison: Source vs Target Domain", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved -> {output_path}")

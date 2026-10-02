"""
Challenge 3: High-Frequency Feature Visualization (Publication Quality)
========================================================================
Grayscale, globally normalized, clean academic styling.
 
Shows high-frequency (style) component activation magnitude:
  - Original: contains edges + domain artifacts
  - After IN: flattened, texture destroyed
  - After CSFD: edges preserved, domain suppressed
"""
 
import argparse
import os
import sys
 
import numpy as np
import torch
import cv2
 
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, transforms as T
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
 
sys.path.insert(0, os.path.abspath("."))
from mask2former import add_maskformer2_config
 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
 
_FEATURES = {}
 
 
def register_feature_hook(model):
    pixel_decoder = model.sem_seg_head.pixel_decoder
    orig_fwd = pixel_decoder.forward_features
    def hooked(features):
        mf, ef, ms = orig_fwd(features)
        _FEATURES['mask_features'] = mf.detach()
        return mf, ef, ms
    pixel_decoder.forward_features = hooked
 
 
def build_cfg(config_file, weights):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.CSFD.ENABLED = True
    cfg.freeze()
    return cfg
 
 
def activation_magnitude(feat):
    """L2 norm across channels → [H, W] raw magnitude (NOT normalized)."""
    if isinstance(feat, torch.Tensor):
        feat = feat.cpu().numpy()
    return np.sqrt((feat ** 2).sum(axis=0))
 
 
def instance_norm_features(feat_tensor):
    B, C, H, W = feat_tensor.shape
    flat = feat_tensor.view(B, C, -1)
    mean = flat.mean(dim=2, keepdim=True)
    std = flat.std(dim=2, keepdim=True) + 1e-6
    return ((flat - mean) / std).view(B, C, H, W)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--csfd-weights", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--num-images", type=int, default=4)
    ap.add_argument("--output-dir", default="experiments/figures/challenge_visuals")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
 
    device = torch.device("cuda")
    cfg = build_cfg(args.config, args.csfd_weights)
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(args.csfd_weights)
    register_feature_hook(model)
 
    csfd_module = model.sem_seg_head.csfd
    low_pass = csfd_module.low_pass
    style_robustifier = csfd_module.style_robustifier
 
    transform = T.ResizeShortestEdge(
        short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
        max_size=cfg.INPUT.MAX_SIZE_TEST, sample_style="choice")
 
    for dataset_name in args.datasets:
        short = dataset_name.replace('_sem_seg_val', '').replace('_fine', '')
        print(f"\n{'='*50}")
        print(f"Dataset: {short}")
        print(f"{'='*50}")
 
        dataset_dicts = DatasetCatalog.get(dataset_name)[:args.num_images]
        all_rows = []
 
        for d_idx, d in enumerate(dataset_dicts):
            bgr = cv2.imread(d['file_name'])
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
 
            transformed = transform.get_transform(bgr).apply_image(bgr)
            image_tensor = torch.as_tensor(
                transformed.astype("float32").transpose(2, 0, 1)).to(device)
 
            _FEATURES.clear()
            with torch.no_grad():
                _ = model([{"image": image_tensor, "height": bgr.shape[0], "width": bgr.shape[1]}])
 
            mask_features = _FEATURES.get('mask_features')
            if mask_features is None:
                continue
 
            x = mask_features
            with torch.no_grad():
                content = low_pass(x)
                style_original = x - content
                style_after_in = instance_norm_features(style_original)
                style_after_csfd = style_robustifier(style_original)
 
            # Raw magnitudes (NOT normalized individually)
            content_mag = activation_magnitude(content[0])
            style_orig_mag = activation_magnitude(style_original[0])
            style_in_mag = activation_magnitude(style_after_in[0])
            style_csfd_mag = activation_magnitude(style_after_csfd[0])
 
            all_rows.append({
                'rgb': rgb,
                'content_mag': content_mag,
                'style_orig_mag': style_orig_mag,
                'style_in_mag': style_in_mag,
                'style_csfd_mag': style_csfd_mag,
            })
            print(f"  Image {d_idx}: processed")
 
        if not all_rows:
            continue
 
        # ---- PER-MAP NORMALIZATION ----
        # Each map normalized independently to show spatial structure clearly.
        # The comparison is about WHERE activations are (edges vs uniform),
        # not absolute magnitude. Magnitude comparison goes in text/table.
        def normalize_map(mag):
            lo, hi = np.percentile(mag, [2, 98])
            return np.clip((mag - lo) / (hi - lo + 1e-8), 0, 1)
 
        # Content also per-map
        normalize_content = normalize_map
        normalize_global = normalize_map
 
        # ---- Create figure (academic style) ----
        n = len(all_rows)
        col_gap = 0.02  # spacing between columns
        row_gap = 0.03  # spacing between rows
 
        fig, axes = plt.subplots(n, 5, figsize=(20, 3.5 * n),
                                  gridspec_kw={'wspace': col_gap, 'hspace': row_gap})
        if n == 1:
            axes = axes[np.newaxis, :]
 
        # Column titles (dark gray, clean)
        col_titles = [
            'Input',
            'Content (low-freq)',
            'Original style (high-freq)',
            'InstanceNorm',
            'CSFD robustifier (ours)',
        ]
 
        for i, row in enumerate(all_rows):
            # Column 0: Input image
            axes[i, 0].imshow(row['rgb'])
            axes[i, 0].axis('off')
            if i == 0:
                axes[i, 0].set_title(col_titles[0], fontsize=12, color='#333333',
                                      fontweight='medium', pad=8)
 
            # Column 1: Content (bone colormap)
            axes[i, 1].imshow(normalize_content(row['content_mag']),
                              cmap='bone', vmin=0, vmax=1)
            axes[i, 1].axis('off')
            if i == 0:
                axes[i, 1].set_title(col_titles[1], fontsize=12, color='#333333',
                                      fontweight='medium', pad=8)
 
            # Column 2: Original style (bone colormap, global norm)
            axes[i, 2].imshow(normalize_global(row['style_orig_mag']),
                              cmap='bone', vmin=0, vmax=1)
            axes[i, 2].axis('off')
            if i == 0:
                axes[i, 2].set_title(col_titles[2], fontsize=12, color='#333333',
                                      fontweight='medium', pad=8)
 
            # Column 3: After IN (bone colormap, global norm)
            axes[i, 3].imshow(normalize_global(row['style_in_mag']),
                              cmap='bone', vmin=0, vmax=1)
            axes[i, 3].axis('off')
            if i == 0:
                axes[i, 3].set_title(col_titles[3], fontsize=12, color='#333333',
                                      fontweight='medium', pad=8)
 
            # Column 4: After CSFD (bone colormap, global norm)
            axes[i, 4].imshow(normalize_global(row['style_csfd_mag']),
                              cmap='bone', vmin=0, vmax=1)
            axes[i, 4].axis('off')
            if i == 0:
                axes[i, 4].set_title(col_titles[4], fontsize=12, color='#333333',
                                      fontweight='medium', pad=8)
 
        plt.tight_layout()
 
        path = os.path.join(args.output_dir, f'challenge3_highfreq_{short}.png')
        plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {path}")
 
    print(f"\nDone. Figures in {args.output_dir}")
 
 
if __name__ == "__main__":
    main()












# """
# Challenge 3: What Happens to Style Information?
# ===============================================
 
# Shows that style contains BOTH harmful (domain stats) AND useful (texture)
# information, and that CSFD preserves the useful part while IN destroys everything.
 
# For each OOD image, visualizes the high-frequency (style) component:
#   1. Original high-freq → rich texture, edges, road markings, AND domain artifacts
#   2. After InstanceNorm → flat, texture destroyed, edges lost
#   3. After CSFD robustifier → edges preserved, road markings retained, domain suppressed
 
# This proves style is not purely harmful and motivates robustification over removal.
 
# Usage:
#     python challenge3_style_vis.py \
#         --config custom_configs/training/csfd_swinb_90k.yaml \
#         --csfd-weights experiments/training_outputs/csfd_swinb_90k/model_final.pth \
#         --datasets acdc_fog_sem_seg_val bdd100k_sem_seg_val \
#         --num-images 4 \
#         --output-dir experiments/figures/challenge_visuals
# """
 
# import argparse
# import os
# import sys
 
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import cv2
 
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.config import get_cfg
# from detectron2.data import DatasetCatalog, transforms as T
# from detectron2.projects.deeplab import add_deeplab_config
# from detectron2.modeling import build_model
 
# sys.path.insert(0, os.path.abspath("."))
# from mask2former import add_maskformer2_config
 
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
 
 
# _FEATURES = {}
 
 
# def register_feature_hook(model):
#     """Capture mask_features from pixel decoder."""
#     pixel_decoder = model.sem_seg_head.pixel_decoder
#     orig_fwd = pixel_decoder.forward_features
#     def hooked(features):
#         mf, ef, ms = orig_fwd(features)
#         _FEATURES['mask_features'] = mf.detach()  # keep on GPU for CSFD processing
#         return mf, ef, ms
#     pixel_decoder.forward_features = hooked
 
 
# def build_cfg(config_file, weights):
#     cfg = get_cfg()
#     add_deeplab_config(cfg)
#     add_maskformer2_config(cfg)
#     cfg.merge_from_file(config_file)
#     cfg.defrost()
#     cfg.MODEL.WEIGHTS = weights
#     cfg.MODEL.CSFD.ENABLED = True
#     cfg.freeze()
#     return cfg
 
 
# def activation_magnitude(feat):
#     """Compute per-pixel activation magnitude: L2 norm across channels.
    
#     Input: [C, H, W] tensor or numpy array
#     Output: [H, W] numpy array (normalized to [0, 1])
#     """
#     if isinstance(feat, torch.Tensor):
#         feat = feat.cpu().numpy()
#     mag = np.sqrt((feat ** 2).sum(axis=0))  # [H, W]
#     # Normalize to [0, 1] using percentile for robustness
#     lo, hi = np.percentile(mag, [2, 98])
#     mag = np.clip((mag - lo) / (hi - lo + 1e-8), 0, 1)
#     return mag
 
 
# def instance_norm_features(feat_tensor):
#     """Apply Instance Normalization to features [1, C, H, W]."""
#     B, C, H, W = feat_tensor.shape
#     flat = feat_tensor.view(B, C, -1)
#     mean = flat.mean(dim=2, keepdim=True)
#     std = flat.std(dim=2, keepdim=True) + 1e-6
#     normed = ((flat - mean) / std).view(B, C, H, W)
#     return normed
 
 
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", required=True)
#     ap.add_argument("--csfd-weights", required=True)
#     ap.add_argument("--datasets", nargs="+", required=True)
#     ap.add_argument("--num-images", type=int, default=4)
#     ap.add_argument("--output-dir", default="experiments/figures/challenge_visuals")
#     args = ap.parse_args()
#     os.makedirs(args.output_dir, exist_ok=True)
 
#     device = torch.device("cuda")
 
#     # Load CSFD model (we need its internal components)
#     cfg = build_cfg(args.config, args.csfd_weights)
#     model = build_model(cfg)
#     model.eval()
#     DetectionCheckpointer(model).load(args.csfd_weights)
#     register_feature_hook(model)
 
#     # Get CSFD module components
#     csfd_module = model.sem_seg_head.csfd
#     low_pass = csfd_module.low_pass
#     style_robustifier = csfd_module.style_robustifier
 
#     transform = T.ResizeShortestEdge(
#         short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
#         max_size=cfg.INPUT.MAX_SIZE_TEST, sample_style="choice")
 
#     for dataset_name in args.datasets:
#         short = dataset_name.replace('_sem_seg_val', '').replace('_fine', '')
#         print(f"\n{'='*50}")
#         print(f"Dataset: {short}")
#         print(f"{'='*50}")
 
#         dataset_dicts = DatasetCatalog.get(dataset_name)[:args.num_images]
#         all_rows = []
 
#         for d_idx, d in enumerate(dataset_dicts):
#             bgr = cv2.imread(d['file_name'])
#             if bgr is None:
#                 continue
#             rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
 
#             # Run forward to get mask_features
#             transformed = transform.get_transform(bgr).apply_image(bgr)
#             image_tensor = torch.as_tensor(
#                 transformed.astype("float32").transpose(2, 0, 1)).to(device)
 
#             _FEATURES.clear()
#             with torch.no_grad():
#                 inp = [{"image": image_tensor, "height": bgr.shape[0], "width": bgr.shape[1]}]
#                 _ = model(inp)
 
#             mask_features = _FEATURES.get('mask_features')
#             if mask_features is None:
#                 continue
 
#             # mask_features: [1, C, H, W]
#             x = mask_features
 
#             with torch.no_grad():
#                 # Step 1: Decompose using CSFD's learned low-pass
#                 content = low_pass(x)                   # [1, C, H, W]
#                 style_original = x - content             # high-freq residual
 
#                 # Step 2: What IN does to the high-freq
#                 style_after_in = instance_norm_features(style_original)
 
#                 # Step 3: What CSFD robustifier does to the high-freq
#                 style_after_csfd = style_robustifier(style_original)
 
#             # Compute activation magnitudes for visualization
#             content_mag = activation_magnitude(content[0])
#             style_orig_mag = activation_magnitude(style_original[0])
#             style_in_mag = activation_magnitude(style_after_in[0])
#             style_csfd_mag = activation_magnitude(style_after_csfd[0])
 
#             all_rows.append({
#                 'rgb': rgb,
#                 'content_mag': content_mag,
#                 'style_orig_mag': style_orig_mag,
#                 'style_in_mag': style_in_mag,
#                 'style_csfd_mag': style_csfd_mag,
#             })
#             print(f"  Image {d_idx}: processed")
 
#         if not all_rows:
#             continue
 
#         # ---- Create figure ----
#         n = len(all_rows)
#         fig, axes = plt.subplots(n, 5, figsize=(22, 4 * n))
#         if n == 1:
#             axes = axes[np.newaxis, :]
 
#         cmap = 'inferno'  # high contrast, clear visualization
 
#         for i, row in enumerate(all_rows):
#             # Column 0: Input image
#             axes[i, 0].imshow(row['rgb'])
#             axes[i, 0].axis('off')
#             if i == 0:
#                 axes[i, 0].set_title('Input Image', fontsize=13, fontweight='bold')
 
#             # Column 1: Content (low-freq) - domain-invariant structure
#             axes[i, 1].imshow(row['content_mag'], cmap=cmap, vmin=0, vmax=1)
#             axes[i, 1].axis('off')
#             if i == 0:
#                 axes[i, 1].set_title('Content\n(Low-frequency)', fontsize=13,
#                                       fontweight='bold', color='#2166ac')
 
#             # Column 2: Original style (high-freq) - contains both useful + harmful
#             axes[i, 2].imshow(row['style_orig_mag'], cmap=cmap, vmin=0, vmax=1)
#             axes[i, 2].axis('off')
#             if i == 0:
#                 axes[i, 2].set_title('Original Style\n(High-frequency)',
#                                       fontsize=13, fontweight='bold', color='#333333')
 
#             # Column 3: After IN - texture destroyed
#             axes[i, 3].imshow(row['style_in_mag'], cmap=cmap, vmin=0, vmax=1)
#             axes[i, 3].axis('off')
#             if i == 0:
#                 axes[i, 3].set_title('After InstanceNorm\n(texture destroyed)',
#                                       fontsize=13, fontweight='bold', color='#cc0000')
#             for spine in axes[i, 3].spines.values():
#                 spine.set_edgecolor('#cc0000')
#                 spine.set_linewidth(3)
#                 spine.set_visible(True)
 
#             # Column 4: After CSFD robustifier - useful texture preserved
#             axes[i, 4].imshow(row['style_csfd_mag'], cmap=cmap, vmin=0, vmax=1)
#             axes[i, 4].axis('off')
#             if i == 0:
#                 axes[i, 4].set_title('After CSFD Robustifier\n(edges & texture preserved)',
#                                       fontsize=13, fontweight='bold', color='#006600')
#             for spine in axes[i, 4].spines.values():
#                 spine.set_edgecolor('#006600')
#                 spine.set_linewidth(3)
#                 spine.set_visible(True)
 
#         # Add column descriptions at bottom
#         fig.text(0.30, -0.01,
#                  'Style features contain edges, road markings, and object boundaries (useful)\n'
#                  'as well as domain-specific illumination and contrast patterns (harmful).',
#                  ha='center', fontsize=10, color='#666666', fontstyle='italic')
#         fig.text(0.70, -0.01,
#                  'InstanceNorm removes everything → edges and texture lost.\n'
#                  'CSFD robustifier suppresses domain statistics while preserving spatial structure.',
#                  ha='center', fontsize=10, color='#666666', fontstyle='italic')
 
#         fig.suptitle(
#             f'Challenge 3: Style Contains Useful Information — {short}\n'
#             f'High-frequency feature activation magnitude (brighter = stronger activation)',
#             fontsize=15, fontweight='bold', y=1.02)
#         plt.tight_layout()
 
#         path = os.path.join(args.output_dir, f'challenge3_style_features_{short}.png')
#         plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
#         plt.close()
#         print(f"  Saved: {path}")
 
#     print(f"\nDone. Figures in {args.output_dir}")
 
 
# if __name__ == "__main__":
#     main()
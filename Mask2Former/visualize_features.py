"""
Feature Quality Visualization (PCA)
====================================
Extracts mask_features from baseline and CSFD models for the same images,
projects to RGB via PCA, saves side-by-side comparisons.
 
PCA is fit on Cityscapes (ID) features. OOD features that are well-aligned
with ID show similar color patterns; corrupted features look different.
 
Usage:
    python visualize_features.py \
        --config custom_configs/training/csfd_swinb_90k.yaml \
        --baseline-weights experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth \
        --csfd-weights experiments/training_outputs/csfd_swinb_90k/model_final.pth \
        --datasets cityscapes_fine_sem_seg_val acdc_fog_sem_seg_val acdc_night_sem_seg_val bdd100k_sem_seg_val \
        --num-images 5 \
        --output-dir experiments/figures/feature_quality
"""
 
import argparse
import os
import sys
 
import numpy as np
import torch
from sklearn.decomposition import PCA
 
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
 
sys.path.insert(0, os.path.abspath("."))
from mask2former import add_maskformer2_config
 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
 
_FEATURES = {}
 
 
def register_feature_hook(model):
    """Hook pixel decoder to capture mask_features before CSFD touches them."""
    pixel_decoder = model.sem_seg_head.pixel_decoder
    orig_fwd = pixel_decoder.forward_features
 
    def hooked(features):
        mask_features, enc_feat, ms_feat = orig_fwd(features)
        _FEATURES['mask_features'] = mask_features.detach().cpu()
        return mask_features, enc_feat, ms_feat
 
    pixel_decoder.forward_features = hooked
 
 
def build_cfg(config_file, weights, csfd_enabled):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.CSFD.ENABLED = csfd_enabled
    cfg.freeze()
    return cfg
 
 
def extract_features(cfg, model, dataset_name, num_images):
    loader = build_detection_test_loader(cfg, dataset_name)
    feats, imgs = [], []
    count = 0
    for batch in loader:
        if count >= num_images:
            break
        _FEATURES.clear()
        with torch.no_grad():
            _ = model(batch)
        f = _FEATURES.get('mask_features')
        if f is not None:
            feats.append(f[0].numpy())
            img = batch[0]['image'].cpu().float().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            imgs.append(img)
        count += 1
    return feats, imgs
 
 
def features_to_rgb(features_list, pca=None, fit=False):
    if pca is None:
        pca = PCA(n_components=3)
    if fit:
        pixels = []
        for f in features_list:
            C, H, W = f.shape
            p = f.reshape(C, -1).T
            idx = np.random.choice(len(p), min(5000, len(p)), replace=False)
            pixels.append(p[idx])
        pca.fit(np.concatenate(pixels))
 
    rgbs = []
    for f in features_list:
        C, H, W = f.shape
        rgb = pca.transform(f.reshape(C, -1).T).reshape(H, W, 3)
        for c in range(3):
            lo, hi = np.percentile(rgb[:, :, c], [2, 98])
            rgb[:, :, c] = np.clip((rgb[:, :, c] - lo) / (hi - lo + 1e-8), 0, 1)
        rgbs.append(rgb)
    return rgbs, pca
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--baseline-weights", required=True)
    ap.add_argument("--csfd-weights", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--num-images", type=int, default=5)
    ap.add_argument("--output-dir", default="experiments/figures/feature_quality")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
 
    # Build both models
    print("Loading baseline model...")
    b_cfg = build_cfg(args.config, args.baseline_weights, csfd_enabled=False)
    b_model = build_model(b_cfg)
    b_model.eval()
    DetectionCheckpointer(b_model).load(args.baseline_weights)
    register_feature_hook(b_model)
 
    print("Loading CSFD model...")
    c_cfg = build_cfg(args.config, args.csfd_weights, csfd_enabled=True)
    c_model = build_model(c_cfg)
    c_model.eval()
    DetectionCheckpointer(c_model).load(args.csfd_weights)
    register_feature_hook(c_model)
 
    pca = None
    for i, ds in enumerate(args.datasets):
        short = ds.replace('_sem_seg_val', '').replace('_fine', '')
        print(f"\n--- {short} ---")
 
        print("  Baseline...")
        b_feats, imgs = extract_features(b_cfg, b_model, ds, args.num_images)
        print("  CSFD...")
        c_feats, _ = extract_features(c_cfg, c_model, ds, args.num_images)
 
        if not b_feats or not c_feats:
            print(f"  Skipping {short}")
            continue
 
        fit = (i == 0)
        if fit:
            print("  Fitting PCA on ID...")
        b_rgb, pca = features_to_rgb(b_feats, pca, fit=fit)
        c_rgb, _ = features_to_rgb(c_feats, pca, fit=False)
 
        n = min(len(imgs), len(b_rgb), len(c_rgb))
        fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
        if n == 1:
            axes = axes[np.newaxis, :]
 
        for j in range(n):
            axes[j, 0].imshow(imgs[j])
            axes[j, 0].axis('off')
            axes[j, 1].imshow(b_rgb[j])
            axes[j, 1].axis('off')
            axes[j, 2].imshow(c_rgb[j])
            axes[j, 2].axis('off')
            if j == 0:
                axes[j, 0].set_title('Input Image', fontsize=13, fontweight='bold')
                axes[j, 1].set_title('Baseline Features (PCA)', fontsize=13, fontweight='bold')
                axes[j, 2].set_title('CSFD Features (PCA)', fontsize=13, fontweight='bold')
 
        fig.suptitle(f'Feature Quality — {short}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(args.output_dir, f'feature_quality_{short}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")
 
    print(f"\nDone. Figures in {args.output_dir}")
 
 
if __name__ == "__main__":
    main()
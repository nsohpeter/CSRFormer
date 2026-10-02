#!/usr/bin/env python3
"""
Per-Class Domain t-SNE
=======================
The global t-SNE (tsne_annotated_v2.py) colors by class AND shapes by
domain in one plot -- but semantic class is a much larger axis of
variation than domain, so t-SNE puts class-separation front and
center and any (real, but smaller) domain-clustering difference is
invisible, buried inside each class's already-tight cluster.

This script isolates the comparison the paper actually needs: for
EACH class separately, pool that class's pixels across all domains,
run t-SNE on THAT SUBSET ONLY (class held fixed), and measure/plot
domain clustering within it. With class no longer competing for
variance, any real domain-clustering difference between baseline and
CSFD should become visible per class, matching what
silhouette_per_class.png already measures quantitatively -- this is
its visual companion, one small t-SNE panel per class per model.

Checkpoint provenance reminder (same as elsewhere in this project):
csfd_swinb_90k/model_final.pth is a confirmed-dead iteration-4
snapshot. Pass csfd_v2_swinb_90k/model_final.pth for --csfd-weights.
Verify --baseline-weights the same way before trusting output:
    python3 -c "import torch; print(torch.load('<path>', map_location='cpu').get('iteration'))"

Usage:
    python perclass_tsne.py \
        --config custom_configs/training/csfd_swinb_90k.yaml \
        --baseline-weights experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth \
        --csfd-weights experiments/training_outputs/csfd_v2_swinb_90k/model_final.pth \
        --datasets cityscapes_fine_sem_seg_val bdd100k_sem_seg_val gta5_sem_seg_val \
        --num-images 20 --samples-per-class 50
"""

import argparse
import os
import sys

import numpy as np
import torch
import cv2
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

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
import matplotlib.lines as mlines


_FEATURES = {}

SELECTED_CLASSES = {0: 'road', 2: 'building', 8: 'vegetation', 10: 'sky', 13: 'car', 11: 'person'}

DOMAIN_COLORS = {'cityscapes': '#6baed6', 'bdd100k': '#fdae6b', 'gta5': '#74c476'}
DOMAIN_LABELS = {'cityscapes': 'Cityscapes', 'bdd100k': 'BDD100K', 'gta5': 'GTA5'}
DOMAIN_MARKERS = {'cityscapes': 'o', 'bdd100k': '^', 'gta5': 's'}


def register_feature_hook(model):
    pixel_decoder = model.sem_seg_head.pixel_decoder
    orig_fwd = pixel_decoder.forward_features
    def hooked(features):
        mf, ef, ms = orig_fwd(features)
        _FEATURES['mask_features'] = mf.detach().cpu()
        return mf, ef, ms
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


def extract_features_with_classes(cfg, model, dataset_name, num_images, samples_per_class=50):
    dataset_dicts = DatasetCatalog.get(dataset_name)[:num_images]
    transform = T.ResizeShortestEdge(
        short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
        max_size=cfg.INPUT.MAX_SIZE_TEST, sample_style="choice")
    class_features = {c: [] for c in SELECTED_CLASSES}

    for d in dataset_dicts:
        gt_path = d.get('sem_seg_file_name')
        if gt_path is None:
            continue
        bgr = cv2.imread(d['file_name'])
        if bgr is None:
            continue
        transformed = transform.get_transform(bgr).apply_image(bgr)
        image_tensor = torch.as_tensor(
            transformed.astype("float32").transpose(2, 0, 1)
        ).to(next(model.parameters()).device)
        inp = [{"image": image_tensor, "height": d['height'], "width": d['width']}]
        _FEATURES.clear()
        with torch.no_grad():
            _ = model(inp)
        feat = _FEATURES.get('mask_features')
        if feat is None:
            continue
        feat = feat[0].numpy()
        C, H, W = feat.shape
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt is None:
            continue
        gt_r = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
        for cls_id in SELECTED_CLASSES:
            if len(class_features[cls_id]) >= samples_per_class * num_images:
                continue
            coords = np.argwhere(gt_r == cls_id)
            if len(coords) == 0:
                continue
            n = min(samples_per_class, len(coords))
            idx = np.random.choice(len(coords), n, replace=False)
            for h, w in coords[idx]:
                class_features[cls_id].append(feat[:, h, w])

    all_f, all_c = [], []
    for c in SELECTED_CLASSES:
        if class_features[c]:
            all_f.extend(class_features[c])
            all_c.extend([c] * len(class_features[c]))
    if not all_f:
        return np.zeros((0, 256)), np.zeros(0, dtype=int)
    return np.stack(all_f), np.array(all_c, dtype=int)


def run_perclass(baseline_data, csfd_data, output_dir, min_samples=20):
    class_ids = sorted(SELECTED_CLASSES.keys())
    n_classes = len(class_ids)

    fig, axes = plt.subplots(2, n_classes, figsize=(3.2 * n_classes, 6.4))
    row_specs = [('Baseline', baseline_data), ('CSFD (ours)', csfd_data)]
    results = []

    for row, (row_label, data_list) in enumerate(row_specs):
        for col, cls_id in enumerate(class_ids):
            ax = axes[row, col]
            feats, doms = [], []
            for f, c, ds in data_list:
                mask = c == cls_id
                if mask.sum() == 0:
                    continue
                feats.append(f[mask])
                doms.extend([ds] * int(mask.sum()))

            if not feats:
                ax.axis('off')
                continue
            feats = np.concatenate(feats)
            doms = np.array(doms)

            if len(feats) < min_samples or len(set(doms)) < 2:
                ax.text(0.5, 0.5, 'insufficient\nsamples', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8, color='#999999')
                ax.set_xticks([]); ax.set_yticks([])
                continue

            perp = min(30, max(5, len(feats) // 4))
            emb = TSNE(n_components=2, perplexity=perp, n_iter=1000,
                       random_state=42).fit_transform(feats)
            sil = silhouette_score(emb, doms)
            results.append((row_label, SELECTED_CLASSES[cls_id], sil, len(feats)))

            for domain_full in sorted(set(doms)):
                ds_short = domain_full.replace('_sem_seg_val', '').replace('_fine', '')
                m = doms == domain_full
                ax.scatter(emb[m, 0], emb[m, 1],
                           c=DOMAIN_COLORS.get(ds_short, '#999'),
                           marker=DOMAIN_MARKERS.get(ds_short, 'o'),
                           s=25, alpha=0.7, edgecolors='black', linewidths=0.3)

            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{SELECTED_CLASSES[cls_id]}\nsil={sil:.3f}", fontsize=10)

        axes[row, 0].set_ylabel(row_label, fontsize=12, fontweight='bold')

    domain_handles = [
        mlines.Line2D([0], [0], marker=DOMAIN_MARKERS[ds], color='w',
                      markerfacecolor=DOMAIN_COLORS[ds], markeredgecolor='black',
                      markersize=8, label=DOMAIN_LABELS[ds])
        for ds in DOMAIN_MARKERS
    ]
    fig.legend(handles=domain_handles, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.06), fontsize=10, frameon=False)

    plt.tight_layout()
    path = os.path.join(output_dir, 'tsne_perclass_domain.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {path}")

    print("\nPer-class domain silhouette (t-SNE, class held fixed):")
    print(f"{'model':<14}{'class':<12}{'silhouette':>12}{'n':>8}")
    for row_label, cname, sil, n in results:
        print(f"{row_label:<14}{cname:<12}{sil:>12.3f}{n:>8d}")

    # Paired comparison, since that's the actual claim: did silhouette
    # go DOWN from Baseline to CSFD for the same class?
    print("\nPaired change (Baseline -> CSFD), negative = improved:")
    b_map = {c: s for r, c, s, n in results if r == 'Baseline'}
    c_map = {c: s for r, c, s, n in results if r == 'CSFD (ours)'}
    for cname in b_map:
        if cname in c_map:
            delta = c_map[cname] - b_map[cname]
            print(f"  {cname:<12} {b_map[cname]:.3f} -> {c_map[cname]:.3f}  "
                  f"({'improved' if delta < 0 else 'worse'}, delta={delta:+.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--baseline-weights", required=True)
    ap.add_argument("--csfd-weights", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--num-images", type=int, default=20)
    ap.add_argument("--samples-per-class", type=int, default=50)
    ap.add_argument("--output-dir", default="experiments/figures/feature_clusters")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print("BASELINE features")
    print("=" * 50)
    b_cfg = build_cfg(args.config, args.baseline_weights, False)
    b_model = build_model(b_cfg)
    b_model.eval()
    DetectionCheckpointer(b_model).load(args.baseline_weights)
    register_feature_hook(b_model)
    baseline_data = []
    for ds in args.datasets:
        short = ds.replace('_sem_seg_val', '').replace('_fine', '')
        print(f"  {short}...")
        f, c = extract_features_with_classes(b_cfg, b_model, ds, args.num_images, args.samples_per_class)
        print(f"    {len(f)} samples")
        if len(f) > 0:
            baseline_data.append((f, c, ds))
    del b_model
    torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("CSFD features")
    print("=" * 50)
    c_cfg = build_cfg(args.config, args.csfd_weights, True)
    c_model = build_model(c_cfg)
    c_model.eval()
    DetectionCheckpointer(c_model).load(args.csfd_weights)
    register_feature_hook(c_model)
    csfd_data = []
    for ds in args.datasets:
        short = ds.replace('_sem_seg_val', '').replace('_fine', '')
        print(f"  {short}...")
        f, c = extract_features_with_classes(c_cfg, c_model, ds, args.num_images, args.samples_per_class)
        print(f"    {len(f)} samples")
        if len(f) > 0:
            csfd_data.append((f, c, ds))
    del c_model
    torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("Per-class t-SNE")
    print("=" * 50)
    if baseline_data and csfd_data:
        run_perclass(baseline_data, csfd_data, args.output_dir)
    else:
        print("ERROR: No data extracted.")


if __name__ == "__main__":
    main()
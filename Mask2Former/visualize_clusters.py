#!/usr/bin/env python3
"""
Annotated Content-Style Disentanglement t-SNE
=============================================
Matches the reference style: red/green titles, dashed cluster circles,
annotation boxes on sides, arrow between panels.
"""
 
import argparse
import os
import sys
 
import numpy as np
import torch
import cv2
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial import ConvexHull
 
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
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyArrowPatch
import matplotlib.lines as mlines
 
 
_FEATURES = {}
 
SELECTED_CLASSES = {0: 'road', 2: 'building', 8: 'vegetation', 10: 'sky', 13: 'car', 11: 'person'}
 
DOMAIN_COLORS = {
    'cityscapes': '#6baed6',
    'bdd100k': '#fdae6b',
    'gta5': '#74c476',
}
 
DOMAIN_LABELS = {
    'cityscapes': 'Cityscapes',
    'bdd100k': 'BDD100K',
    'gta5': 'GTA5',
}
 
 
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
 
 
def draw_domain_ellipses(ax, embeddings, domains, alpha=0.15):
    """Draw dashed ellipses around each domain's point cloud."""
    ellipse_artists = []
    for domain in sorted(set(domains)):
        domain_short = domain.replace('_sem_seg_val', '').replace('_fine', '')
        mask = domains == domain
        pts = embeddings[mask]
        if len(pts) < 5:
            continue
 
        # Fit ellipse using mean and covariance
        mean = pts.mean(axis=0)
        cov = np.cov(pts.T)
 
        # Eigenvalues for ellipse axes
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
 
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        # 2.0 std covers ~95% of points
        width = 2.0 * 2 * np.sqrt(eigenvalues[0])
        height = 2.0 * 2 * np.sqrt(eigenvalues[1])
 
        color = DOMAIN_COLORS.get(domain_short, '#999999')
 
        ellipse = Ellipse(
            xy=mean, width=width, height=height, angle=angle,
            fill=False, edgecolor=color, linewidth=2.0,
            linestyle='--', alpha=0.7
        )
        ax.add_patch(ellipse)
 
        # Light fill
        ellipse_fill = Ellipse(
            xy=mean, width=width, height=height, angle=angle,
            fill=True, facecolor=color, alpha=0.06,
            edgecolor='none'
        )
        ax.add_patch(ellipse_fill)
 
    return ellipse_artists
 
 
def add_annotation_box(ax, text, xy, xytext, color='#333333', fontsize=9):
    """Add an annotation box with arrow."""
    ax.annotate(
        text, xy=xy, xytext=xytext,
        textcoords='axes fraction', 
        fontsize=fontsize,
        ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0',
                  edgecolor=color, alpha=0.9, linewidth=1.5),
        arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
    )
 
 
def create_annotated_figure(baseline_data, csfd_data, output_dir):
    """Create the annotated t-SNE figure matching reference style."""
 
    def prepare(data_list):
        all_f = np.concatenate([d[0] for d in data_list])
        all_c = np.concatenate([d[1] for d in data_list])
        all_d = []
        for f, c, d in data_list:
            all_d.extend([d] * len(f))
        all_d = np.array(all_d)
        max_pts = 6000
        if len(all_f) > max_pts:
            np.random.seed(42)
            idx = np.random.choice(len(all_f), max_pts, replace=False)
            all_f, all_c, all_d = all_f[idx], all_c[idx], all_d[idx]
        return all_f, all_c, all_d
 
    b_f, b_c, b_d = prepare(baseline_data)
    c_f, c_c, c_d = prepare(csfd_data)
 
    print("  t-SNE (baseline)...")
    emb_b = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42).fit_transform(b_f)
    print("  t-SNE (CSFD)...")
    emb_c = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42).fit_transform(c_f)
 
    # Silhouette scores
    b_sil = silhouette_score(emb_b, b_d)
    c_sil = silhouette_score(emb_c, c_d)
    print(f"  Baseline silhouette: {b_sil:.3f}")
    print(f"  CSFD silhouette:     {c_sil:.3f}")
 
    # ---- Create figure ----
    fig = plt.figure(figsize=(22, 9))
 
    # Layout: [left_annotations | baseline_plot | arrow | csfd_plot | right_annotations]
    gs = fig.add_gridspec(1, 5, width_ratios=[0.12, 1, 0.06, 1, 0.12],
                          wspace=0.02)
 
    ax_left_text = fig.add_subplot(gs[0, 0])
    ax_baseline = fig.add_subplot(gs[0, 1])
    ax_arrow = fig.add_subplot(gs[0, 2])
    ax_csfd = fig.add_subplot(gs[0, 3])
    ax_right_text = fig.add_subplot(gs[0, 4])
 
    # Hide text and arrow axes
    for a in [ax_left_text, ax_arrow, ax_right_text]:
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.axis('off')
 
    # ---- Plot baseline ----
    for domain_full in sorted(set(b_d)):
        ds = domain_full.replace('_sem_seg_val', '').replace('_fine', '')
        mask = b_d == domain_full
        ax_baseline.scatter(emb_b[mask, 0], emb_b[mask, 1],
                           c=DOMAIN_COLORS.get(ds, '#999'),
                           s=12, alpha=0.45, edgecolors='none',
                           label=DOMAIN_LABELS.get(ds, ds))
 
    draw_domain_ellipses(ax_baseline, emb_b, b_d)
    ax_baseline.set_xticks([])
    ax_baseline.set_yticks([])
    ax_baseline.set_title('Baseline Mask2Former (Style-Entangled)',
                          fontsize=14, fontweight='bold', color='#cc0000', pad=12)
    ax_baseline.text(0.5, -0.02, 'Feature Space (Pixel Decoder Output)',
                     transform=ax_baseline.transAxes, ha='center', fontsize=10,
                     color='#666666')
    ax_baseline.legend(fontsize=10, loc='upper right', framealpha=0.9,
                       markerscale=2.5)
 
    # ---- Plot CSFD ----
    for domain_full in sorted(set(c_d)):
        ds = domain_full.replace('_sem_seg_val', '').replace('_fine', '')
        mask = c_d == domain_full
        ax_csfd.scatter(emb_c[mask, 0], emb_c[mask, 1],
                       c=DOMAIN_COLORS.get(ds, '#999'),
                       s=12, alpha=0.45, edgecolors='none',
                       label=DOMAIN_LABELS.get(ds, ds))
 
    draw_domain_ellipses(ax_csfd, emb_c, c_d)
    ax_csfd.set_xticks([])
    ax_csfd.set_yticks([])
    ax_csfd.set_title('CSFD (Ours) (More Domain-Invariant)',
                      fontsize=14, fontweight='bold', color='#006600', pad=12)
    ax_csfd.text(0.5, -0.02, 'Feature Space (Pixel Decoder Output)',
                 transform=ax_csfd.transAxes, ha='center', fontsize=10,
                 color='#666666')
    ax_csfd.legend(fontsize=10, loc='upper right', framealpha=0.9,
                    markerscale=2.5)
 
    # ---- Arrow between panels ----
    ax_arrow.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                      arrowprops=dict(arrowstyle='->', color='#666666',
                                      lw=3, mutation_scale=25))
 
    # ---- Left annotations (baseline problems) ----
    box_style = dict(boxstyle='round,pad=0.6', linewidth=1.5, alpha=0.95)
 
    ax_left_text.text(0.5, 0.78,
                      'Strong domain\nclustering',
                      ha='center', va='center', fontsize=10,
                      fontweight='bold', color='#cc0000',
                      bbox=dict(facecolor='#fff0f0', edgecolor='#cc0000', **box_style),
                      transform=ax_left_text.transAxes)
 
    ax_left_text.text(0.5, 0.50,
                      'Content and style\nentangled in\nfeature space',
                      ha='center', va='center', fontsize=9,
                      color='#333333',
                      bbox=dict(facecolor='#fff5f0', edgecolor='#cc6666', **box_style),
                      transform=ax_left_text.transAxes)
 
    ax_left_text.text(0.5, 0.22,
                      f'Domain Silhouette\n= {b_sil:.3f}\n(higher = entangled)',
                      ha='center', va='center', fontsize=9,
                      color='#cc0000', fontweight='bold',
                      bbox=dict(facecolor='#fff0f0', edgecolor='#cc0000', **box_style),
                      transform=ax_left_text.transAxes)
 
    # ---- Right annotations (CSFD improvements) ----
    ax_right_text.text(0.5, 0.78,
                       'Weak domain\nclustering',
                       ha='center', va='center', fontsize=10,
                       fontweight='bold', color='#006600',
                       bbox=dict(facecolor='#f0fff0', edgecolor='#006600', **box_style),
                       transform=ax_right_text.transAxes)
 
    ax_right_text.text(0.5, 0.50,
                       'Style robustified\nand fused back;\nsemantics preserved',
                       ha='center', va='center', fontsize=9,
                       color='#333333',
                       bbox=dict(facecolor='#f0fff5', edgecolor='#66cc66', **box_style),
                       transform=ax_right_text.transAxes)
 
    ax_right_text.text(0.5, 0.22,
                       f'Domain Silhouette\n= {c_sil:.3f}\n(lower = disentangled)',
                       ha='center', va='center', fontsize=9,
                       color='#006600', fontweight='bold',
                       bbox=dict(facecolor='#f0fff0', edgecolor='#006600', **box_style),
                       transform=ax_right_text.transAxes)
 
    # ---- Main title ----
    fig.suptitle('Feature Space Visualization: Content-Style Disentanglement in mask_features',
                 fontsize=16, fontweight='bold', y=0.98)
 
    plt.tight_layout(rect=[0, 0, 1, 0.95])
 
    path = os.path.join(output_dir, 'tsne_annotated.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")
 
    # ---- Also save the silhouette bar chart ----
    # Per-class silhouette
    fig2, ax2 = plt.subplots(figsize=(10, 5))
 
    class_ids = sorted(SELECTED_CLASSES.keys())
    b_class_sil, c_class_sil = {}, {}
    for cls_id in class_ids:
        for emb, doms, classes, store in [(emb_b, b_d, b_c, b_class_sil),
                                           (emb_c, c_d, c_c, c_class_sil)]:
            mask = classes == cls_id
            if mask.sum() >= 10:
                cls_doms = doms[mask]
                if len(set(cls_doms)) >= 2:
                    store[cls_id] = silhouette_score(emb[mask], cls_doms)
 
    valid_ids = [c for c in class_ids if c in b_class_sil and c in c_class_sil]
    x = np.arange(len(valid_ids))
    w = 0.35
 
    b_v = [b_class_sil[c] for c in valid_ids]
    c_v = [c_class_sil[c] for c in valid_ids]
    names = [SELECTED_CLASSES[c] for c in valid_ids]
 
    bars1 = ax2.bar(x - w/2, b_v, w, label='Baseline M2F', color='#cc4444', alpha=0.8)
    bars2 = ax2.bar(x + w/2, c_v, w, label='CSFD (ours)', color='#44aa44', alpha=0.8)
 
    for bar, v in zip(bars1, b_v):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=9, color='#cc4444',
                 fontweight='bold')
    for bar, v in zip(bars2, c_v):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=9, color='#44aa44',
                 fontweight='bold')
 
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=11)
    ax2.set_ylabel('Domain Silhouette Score\n(lower = more disentangled)', fontsize=11)
    ax2.set_title('Per-Class Domain Separation: Baseline vs CSFD',
                   fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
 
    path2 = os.path.join(output_dir, 'silhouette_per_class.png')
    plt.savefig(path2, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path2}")
 
 
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
 
    # Baseline
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
 
    # CSFD
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
 
    # Create figures
    print("\n" + "=" * 50)
    print("Creating figures")
    print("=" * 50)
    if baseline_data and csfd_data:
        create_annotated_figure(baseline_data, csfd_data, args.output_dir)
    else:
        print("ERROR: No data extracted.")
    print(f"\nDone. Figures in {args.output_dir}")
 
 
if __name__ == "__main__":
    main()
 
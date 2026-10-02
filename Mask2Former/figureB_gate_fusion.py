"""
Figure B: Adaptive Gate Fusion
================================
Question: does the gate spatially control where the robustified
residual contributes, and is that better/more informative than a
uniform fusion?

Columns: Input | F (raw) | Uniform fusion C+S_r (PCA) | G (gate heatmap) |
         Gated fusion F_hat = C + G*S_r (PCA) | |F_hat - Uniform| (diff)

Design note on capturing G: earlier versions of this script recovered G
algebraically from F_hat, C, and S_r (least-squares per pixel), to avoid
guessing the gate submodule's name. That recovery divides by sum(S_r^2)
per pixel, which is fragile wherever S_r's magnitude is small -- and in
practice produced unstable, spatially-random-looking output (confirmed
against the real csfd_module.py source: NOT what the true gate looks
like). Now that the real source is available, csfd_module.gate (a
SpatialGate instance, called as self.gate(content, style_robust) inside
CSFD.forward) is hooked directly, same pattern as the pixel-decoder and
CSFD-module hooks. This is the actual gate tensor, not an inference.

Also computes three quantitative companions, since whole-image Pearson
correlation between G and Sobel edges came back near-zero (0.001-0.12)
despite G visually tracing object contours clearly -- Sobel fires on
every texture/contrast edge (brick, foliage, clothing folds), so the
vast majority of "edge" pixels are irrelevant clutter that dilutes a
linear correlation even when G's *high* values line up well with real
contours:
  1. Pearson vs Sobel (the original, diluted metric -- kept for
     comparison so the improvement from 2/3 is visible).
  2. Top-decile enrichment vs Sobel: does G's top 10% concentrate on
     Sobel's top 10%, relative to chance? Not diluted by low-signal
     pixels, since it only asks about G's most confident region.
  3. Enrichment AND Pearson vs ground-truth semantic-class-boundary
     masks (dilated class transitions from sem_seg_file_name) instead
     of raw pixel contrast -- the more rigorous reference, since it
     tests against boundaries that are actually semantically
     meaningful rather than any high-contrast pixel.

Usage:
    python figureB_gate_fusion.py \
        --config custom_configs/training/csfd_swinb_90k.yaml \
        --csfd-weights experiments/training_outputs/csfd_swinb_90k/model_final.pth \
        --datasets cityscapes_fine_sem_seg_val bdd100k_sem_seg_val gta5_sem_seg_val acdc_night_sem_seg_val \
        --image-index 0 \
        --num-images 30
"""

import argparse
import os
import sys

import numpy as np
import torch
import cv2

from sklearn.decomposition import PCA
from scipy.stats import pearsonr

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


def register_hooks(model):
    pixel_decoder = model.sem_seg_head.pixel_decoder
    orig_pd_fwd = pixel_decoder.forward_features
    def hooked_pd(features):
        mf, ef, ms = orig_pd_fwd(features)
        _FEATURES['mask_features'] = mf.detach()
        return mf, ef, ms
    pixel_decoder.forward_features = hooked_pd

    csfd_module = model.sem_seg_head.csfd
    orig_csfd_fwd = csfd_module.forward
    def hooked_csfd(mask_features):
        out = orig_csfd_fwd(mask_features)
        _FEATURES['csfd_output'] = out.detach()
        return out
    csfd_module.forward = hooked_csfd

    # Real gate submodule (SpatialGate), hooked directly -- confirmed via
    # actual csfd_module.py source. No more algebraic recovery/division.
    gate_module = csfd_module.gate
    orig_gate_fwd = gate_module.forward
    def hooked_gate(content, style_robust):
        g = orig_gate_fwd(content, style_robust)
        _FEATURES['gate'] = g.detach()
        return g
    gate_module.forward = hooked_gate

    return csfd_module


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


def to_np(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t


def pca_independent(feat_chw, seed=0):
    feat_chw = to_np(feat_chw)
    C, H, W = feat_chw.shape
    flat = feat_chw.reshape(C, -1).T
    pca = PCA(n_components=3, random_state=seed)
    proj = pca.fit_transform(flat).reshape(H, W, 3)
    out = np.zeros_like(proj)
    for c in range(3):
        lo, hi = np.percentile(proj[..., c], [2, 98])
        out[..., c] = np.clip((proj[..., c] - lo) / (hi - lo + 1e-8), 0, 1)
    return out


def pca_fit_reference(feat_chw, seed=0):
    feat_chw = to_np(feat_chw)
    C, H, W = feat_chw.shape
    flat = feat_chw.reshape(C, -1).T
    pca = PCA(n_components=3, random_state=seed).fit(flat)
    proj = pca.transform(flat).reshape(H, W, 3)
    lo_hi = [np.percentile(proj[..., c], [2, 98]) for c in range(3)]
    out = np.zeros_like(proj)
    for c in range(3):
        lo, hi = lo_hi[c]
        out[..., c] = np.clip((proj[..., c] - lo) / (hi - lo + 1e-8), 0, 1)
    return out, pca, lo_hi, (H, W)


def pca_apply_reference(feat_chw, pca, lo_hi, hw):
    feat_chw = to_np(feat_chw)
    C, H, W = feat_chw.shape
    assert (H, W) == hw
    flat = feat_chw.reshape(C, -1).T
    proj = pca.transform(flat).reshape(H, W, 3)
    out = np.zeros_like(proj)
    for c in range(3):
        lo, hi = lo_hi[c]
        out[..., c] = np.clip((proj[..., c] - lo) / (hi - lo + 1e-8), 0, 1)
    return out


def sobel_edge_map(bgr, target_hw):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag_small = cv2.resize(mag, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
    return mag_small


def top_decile_enrichment(G, other, percentile=90, other_is_binary=False):
    """How much does G's top decile concentrate on 'other's high-signal
    region, relative to chance? Fixes the Pearson-correlation dilution
    problem: Pearson is dragged toward 0 by every low-texture,
    low-edge pixel in the image (the vast majority), even when G's
    small set of *high* values line up well with edges/boundaries.
    This instead asks: of the pixels G considers most important, what
    fraction are also edge/boundary pixels, compared to the base rate
    you'd expect from a random selection of the same size?

    Returns an enrichment ratio: 1.0 = no relationship (chance level),
    >1 = G's high-value region is enriched for the reference signal,
    <1 = actively avoids it.
    """
    g_thresh = np.percentile(G, percentile)
    g_top = G >= g_thresh
    if other_is_binary:
        o_top = other.astype(bool)
    else:
        o_thresh = np.percentile(other, percentile)
        o_top = other >= o_thresh
    base_rate = o_top.mean() + 1e-8
    if g_top.mean() < 1e-8:
        return float('nan')
    overlap_rate = (g_top & o_top).mean() / g_top.mean()
    return overlap_rate / base_rate


def gt_boundary_mask(dataset_dict, target_hw, ignore_label=255, dilate_iters=1):
    """Dilated class-transition boundary mask from ground-truth semantic
    labels, resized (nearest-neighbor, to avoid inventing label values)
    to the gate's spatial resolution. Returns None if this dataset dict
    has no semantic label file (caller should skip the GT-based metric
    for that domain rather than fail).

    NOTE: ignore_label=255 is the Cityscapes-style convention. If any
    of your registered datasets use a different ignore value, this
    will silently treat ignore-region transitions as real boundaries
    for that domain -- worth checking register_custom_datasets.py if
    a domain's GT-enrichment number looks anomalously high, since
    ignore regions are often large contiguous blocks whose edges are
    NOT meaningful semantic boundaries.
    """
    label_path = dataset_dict.get('sem_seg_file_name')
    if label_path is None:
        return None
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    if label is None:
        return None
    label_small = cv2.resize(label, (target_hw[1], target_hw[0]),
                              interpolation=cv2.INTER_NEAREST)

    boundary = np.zeros_like(label_small, dtype=bool)
    diff_x = label_small[:, 1:] != label_small[:, :-1]
    diff_y = label_small[1:, :] != label_small[:-1, :]
    boundary[:, :-1] |= diff_x
    boundary[:, 1:] |= diff_x
    boundary[:-1, :] |= diff_y
    boundary[1:, :] |= diff_y

    valid = label_small != ignore_label
    boundary &= valid  # drop boundaries touching ignore regions

    if dilate_iters > 0:
        kernel = np.ones((3, 3), np.uint8)
        boundary = cv2.dilate(boundary.astype(np.uint8), kernel,
                               iterations=dilate_iters).astype(bool)
    return boundary


def run_one_image(model, low_pass, style_robustifier, transform, d, device):
    bgr = cv2.imread(d['file_name'])
    if bgr is None:
        return None
    transformed = transform.get_transform(bgr).apply_image(bgr)
    image_tensor = torch.as_tensor(
        transformed.astype("float32").transpose(2, 0, 1)).to(device)

    _FEATURES.clear()
    with torch.no_grad():
        _ = model([{"image": image_tensor, "height": bgr.shape[0], "width": bgr.shape[1]}])

    mask_features = _FEATURES.get('mask_features')
    csfd_output = _FEATURES.get('csfd_output')
    if mask_features is None or csfd_output is None:
        return None

    with torch.no_grad():
        content = low_pass(mask_features)
        style_original = mask_features - content
        style_robust = style_robustifier(style_original)

    return {
        'bgr': bgr,
        'F': mask_features[0],
        'content': content[0], 'style_robust': style_robust[0],
        'fused': csfd_output[0],
        'gate': _FEATURES['gate'][0, 0],  # [1,H,W] -> [H,W], real captured gate
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--csfd-weights", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--image-index", type=int, default=0)
    ap.add_argument("--num-images", type=int, default=30,
                     help="Images per domain for the gate-vs-edge correlation stat.")
    ap.add_argument("--output-dir", default="experiments/figures/figureB_gate_fusion")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda")
    cfg = build_cfg(args.config, args.csfd_weights)
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(args.csfd_weights)
    csfd_module = register_hooks(model)
    low_pass = csfd_module.low_pass
    style_robustifier = csfd_module.style_robustifier

    transform = T.ResizeShortestEdge(
        short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
        max_size=cfg.INPUT.MAX_SIZE_TEST, sample_style="choice")

    rows = []
    row_labels = []
    corr_results = []

    for dataset_name in args.datasets:
        short = dataset_name.replace('_sem_seg_val', '').replace('_fine', '')
        print(f"Processing {short} ...")
        dataset_dicts = DatasetCatalog.get(dataset_name)

        # ---- quantitative: gate vs edge/boundary, three ways ----
        # 1) Pearson vs Sobel edges (whole-image linear correlation --
        #    diluted by the vast majority of low-texture pixels)
        # 2) Top-decile enrichment vs Sobel edges (does G's high-value
        #    region concentrate on edges, regardless of the rest?)
        # 3) Top-decile enrichment + Pearson vs GT semantic boundaries
        #    (the more rigorous version: real class-transition
        #    boundaries, not just raw pixel contrast)
        corrs, enrich_sobel_list = [], []
        enrich_gt_list, corr_gt_list = [], []
        n_for_corr = min(args.num_images, len(dataset_dicts))
        for k in range(n_for_corr):
            out = run_one_image(model, low_pass, style_robustifier, transform,
                                 dataset_dicts[k], device)
            if out is None:
                continue
            G = to_np(out['gate'])
            edges = sobel_edge_map(out['bgr'], G.shape)
            if edges.std() < 1e-6 or np.std(G) < 1e-8:
                continue
            r, _ = pearsonr(G.flatten(), edges.flatten())
            corrs.append(r)
            enrich_sobel_list.append(top_decile_enrichment(G, edges, percentile=90))

            boundary = gt_boundary_mask(dataset_dicts[k], G.shape)
            if boundary is not None and boundary.any() and boundary.mean() < 1.0:
                enrich_gt_list.append(
                    top_decile_enrichment(G, boundary, percentile=90, other_is_binary=True))
                r_gt, _ = pearsonr(G.flatten(), boundary.flatten().astype(float))
                corr_gt_list.append(r_gt)

        mean_r = float(np.mean(corrs)) if corrs else float('nan')
        mean_enrich_sobel = float(np.nanmean(enrich_sobel_list)) if enrich_sobel_list else float('nan')
        mean_enrich_gt = float(np.nanmean(enrich_gt_list)) if enrich_gt_list else float('nan')
        mean_corr_gt = float(np.mean(corr_gt_list)) if corr_gt_list else float('nan')

        corr_results.append((short, mean_r, mean_enrich_sobel, mean_enrich_gt,
                              mean_corr_gt, len(corrs), len(corr_gt_list)))
        print(f"  Pearson vs Sobel:       {mean_r:.3f}")
        print(f"  Enrichment vs Sobel:    {mean_enrich_sobel:.2f}x chance")
        print(f"  Enrichment vs GT bound: {mean_enrich_gt:.2f}x chance  (n={len(corr_gt_list)})")
        print(f"  Pearson vs GT boundary: {mean_corr_gt:.3f}")

        # ---- qualitative: single representative image ----
        d = dataset_dicts[args.image_index]
        out = run_one_image(model, low_pass, style_robustifier, transform, d, device)
        if out is None:
            continue
        rgb = cv2.cvtColor(out['bgr'], cv2.COLOR_BGR2RGB)
        G = to_np(out['gate'])
        print(f"  G stats: min={G.min():.4f} max={G.max():.4f} "
              f"mean={G.mean():.4f} std={G.std():.4f}")

        F_rgb = pca_independent(out['F'])

        uniform = to_np(out['content']) + to_np(out['style_robust'])  # C + S_r
        gated = to_np(out['fused'])                                    # C + G*S_r

        uni_rgb, uni_pca, uni_lo_hi, hw = pca_fit_reference(uniform)
        gated_rgb = pca_apply_reference(gated, uni_pca, uni_lo_hi, hw)

        diff = gated - uniform  # exactly (G-1)*S_r
        diff_mag = np.sqrt((diff ** 2).sum(axis=0))
        lo, hi = np.percentile(diff_mag, [2, 98])
        diff_norm = np.clip((diff_mag - lo) / (hi - lo + 1e-8), 0, 1)

        # G: per-image percentile normalization, NOT fixed 0-1 -- a fixed
        # scale hides real variation when G's actual range is narrow
        # (e.g. 0.3-0.5), which is expected given gate_mean varies only
        # 0.29-0.59 across domains in prior diagnostics.
        g_lo, g_hi = np.percentile(G, [2, 98])
        G_norm = np.clip((G - g_lo) / (g_hi - g_lo + 1e-8), 0, 1)

        rows.append({'rgb': rgb, 'F': F_rgb, 'G': G_norm, 'uniform': uni_rgb,
                      'gated': gated_rgb, 'diff': diff_norm})
        row_labels.append(short)

    if not rows:
        print("No rows produced.")
        return

    n = len(rows)
    col_keys = ['rgb', 'F', 'uniform', 'G', 'gated', 'diff']
    col_titles = ['Input', r'$F$ (raw)', r'$C+S_r$ (uniform)', r'$G$ (gate)',
                  r'Gated $\hat F=C+G\odot S_r$', r'$\hat F-(C+S_r)$']

    fig, axes = plt.subplots(n, 6, figsize=(20, 2.1 * n),
                              gridspec_kw={'wspace': 0.03, 'hspace': 0.02})
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, row in enumerate(rows):
        for j, key in enumerate(col_keys):
            if key == 'G':
                axes[i, j].imshow(row[key], cmap='viridis', vmin=0, vmax=1, aspect='auto')
            elif key == 'diff':
                axes[i, j].imshow(row[key], cmap='magma', vmin=0, vmax=1, aspect='auto')
            else:
                axes[i, j].imshow(row[key], aspect='auto')
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=13, color='#222222', pad=8)
        axes[i, 0].text(-0.08, 0.5, row_labels[i], transform=axes[i, 0].transAxes,
                         fontsize=11, va='center', ha='right', rotation=90)

    plt.tight_layout()
    path = os.path.join(args.output_dir, 'figureB_gate_fusion.png')
    plt.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {path}")

    print("\n" + "=" * 78)
    print(f"{'domain':<13}{'Pearson/Sobel':>14}{'Enrich/Sobel':>14}"
          f"{'Enrich/GTbound':>15}{'Pearson/GTbound':>16}{'n':>6}")
    for short, r, e_sobel, e_gt, r_gt, n_imgs, n_gt in corr_results:
        print(f"{short:<13}{r:>14.3f}{e_sobel:>13.2f}x{e_gt:>14.2f}x{r_gt:>16.3f}{n_imgs:>6d}")
    print("\nPearson: whole-image linear correlation (diluted by low-texture pixels).")
    print("Enrichment: how much more likely G's top 10% is to land on the")
    print("  reference signal's top 10%, vs. chance (1.0x = no relationship).")
    print("GT boundary: dilated class-transition boundaries from ground truth,")
    print("  not raw pixel contrast -- the more rigorous of the two references.")


if __name__ == "__main__":
    main()
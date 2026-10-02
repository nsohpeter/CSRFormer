"""
Figure A v3: Style Robustification — 6-column PCA false-color visualization
=============================================================================
Input | F (raw) | C (content-oriented) | S (style residual) |
S_r (robustified residual) | ||S_r - S||

Changes vs. v2: dropped the S_n (InstanceNorm) column entirely -- it
was an intermediate diagnostic step, not part of the paper's causal
chain (F -> C/S -> S_r). Row spacing also fixed: the wide vertical
gaps in v2 weren't from hspace, they were imshow's default aspect='equal'
letterboxing each panel to preserve pixel aspect ratio inside a box
sized for a different aspect. aspect='auto' on every imshow call fills
each panel fully instead.

F and C each get their own independent PCA fit+range (they're
structurally different signals -- low-freq content vs. raw feature --
so independent normalization is the honest choice there, same
reasoning as the earlier entangled/disentangled figure).

But S and S_r are two PROCESSING STAGES of the same starting signal,
and the whole point of this figure is to show how far robustification
moves that signal from where it started. Giving each its own
independent PCA fit + percentile stretch would silently erase that --
two signals with very different actual magnitude/structure can both
get independently re-stretched to look equally "busy" once each is
normalized to fill its own full dynamic range.

Fix (unchanged from v2): fit ONE PCA basis on S, and reuse that same
basis + the same clip range (derived from S) for S_r. If S_r is
actually more compressed/consistent than S, it will visibly look more
muted rather than independently re-stretched back to full saturation.

Usage:
    python figureA_robustification.py \
        --config custom_configs/training/csfd_swinb_90k.yaml \
        --csfd-weights experiments/training_outputs/csfd_v2_swinb_90k/model_final.pth \
        --datasets cityscapes_fine_sem_seg_val bdd100k_sem_seg_val gta5_sem_seg_val acdc_night_sem_seg_val \
        --image-index 0
"""

import argparse
import os
import sys

import numpy as np
import torch
import cv2

from sklearn.decomposition import PCA

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


def to_numpy_chw(feat_chw):
    if isinstance(feat_chw, torch.Tensor):
        feat_chw = feat_chw.detach().cpu().numpy()
    return feat_chw


def magnitude_bone(feat_chw):
    """L2-norm-across-channels magnitude map, percentile-normalized.
    Used only for Content (C): its low-pass nature means the magnitude
    map alone already reads as clean scene structure (this is the look
    from the earlier grayscale reference figure) -- PCA color added
    nothing useful here and made it blurrier, not clearer."""
    feat_chw = to_numpy_chw(feat_chw)
    mag = np.sqrt((feat_chw ** 2).sum(axis=0))
    lo, hi = np.percentile(mag, [2, 98])
    return np.clip((mag - lo) / (hi - lo + 1e-8), 0, 1)


def pca_independent(feat_chw, seed=0):
    """Independent PCA fit + independent percentile stretch.
    Use for signals that are NOT stages of the same pipeline (F, C)."""
    feat_chw = to_numpy_chw(feat_chw)
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
    """Fit PCA + percentile range on a REFERENCE signal (S), to be
    reused (not refit) on downstream stages of the same pipeline."""
    feat_chw = to_numpy_chw(feat_chw)
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
    """Apply an ALREADY-FIT PCA basis + range from pca_fit_reference to
    a different signal (S_n or S_r), so relative magnitude/structure
    differences remain visible instead of being independently
    re-stretched away."""
    feat_chw = to_numpy_chw(feat_chw)
    C, H, W = feat_chw.shape
    assert (H, W) == hw, "spatial size mismatch between S and downstream stage"
    flat = feat_chw.reshape(C, -1).T
    proj = pca.transform(flat).reshape(H, W, 3)
    out = np.zeros_like(proj)
    for c in range(3):
        lo, hi = lo_hi[c]
        out[..., c] = np.clip((proj[..., c] - lo) / (hi - lo + 1e-8), 0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--csfd-weights", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--image-index", type=int, default=0)
    ap.add_argument("--output-dir", default="experiments/figures/figureA_robustification")
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

    rows = []
    row_labels = []

    for dataset_name in args.datasets:
        short = dataset_name.replace('_sem_seg_val', '').replace('_fine', '')
        print(f"Processing {short} ...")

        dataset_dicts = DatasetCatalog.get(dataset_name)
        d = dataset_dicts[args.image_index]

        bgr = cv2.imread(d['file_name'])
        if bgr is None:
            print(f"  WARNING: could not read {d['file_name']}, skipping")
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
            print(f"  WARNING: no mask_features captured for {short}, skipping")
            continue

        x = mask_features  # F
        with torch.no_grad():
            content = low_pass(x)                       # C
            style_original = x - content                 # S
            style_robust = style_robustifier(style_original)   # S_r

        F_rgb = pca_independent(x[0])
        C_mag = magnitude_bone(content[0])
        S_rgb, S_pca, S_lo_hi, hw = pca_fit_reference(style_original[0])
        Sr_rgb = pca_apply_reference(style_robust[0], S_pca, S_lo_hi, hw)

        # Explicit diff-magnitude: ||S_r - S|| per pixel (L2 across channels).
        # This is what actually answers "where/how much did robustification
        # change", rather than asking the eye to spot it in two PCA panels.
        diff = to_numpy_chw(style_robust[0]) - to_numpy_chw(style_original[0])
        diff_mag = np.sqrt((diff ** 2).sum(axis=0))
        lo, hi = np.percentile(diff_mag, [2, 98])
        diff_mag_norm = np.clip((diff_mag - lo) / (hi - lo + 1e-8), 0, 1)

        rows.append({
            'rgb': rgb, 'F': F_rgb, 'C': C_mag,
            'S': S_rgb, 'S_r': Sr_rgb,
            'diff': diff_mag_norm,
        })
        row_labels.append(short)
        print("  done")

    if not rows:
        print("No rows produced -- check dataset names / image index.")
        return

    n = len(rows)
    col_keys = ['rgb', 'F', 'C', 'S', 'S_r', 'diff']
    col_titles = ['Input', r'$F$ (raw)', r'$C$ (content-oriented)',
                  r'$S$ (style residual)',
                  r'$S_r$ (robustified residual)', r'$\Vert S_r - S\Vert$']

    fig, axes = plt.subplots(n, 6, figsize=(20, 2.1 * n),
                              gridspec_kw={'wspace': 0.03, 'hspace': 0.02})
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, row in enumerate(rows):
        for j, key in enumerate(col_keys):
            if key == 'diff':
                axes[i, j].imshow(row[key], cmap='magma', vmin=0, vmax=1, aspect='auto')
            elif key == 'C':
                axes[i, j].imshow(row[key], cmap='bone', vmin=0, vmax=1, aspect='auto')
            else:
                axes[i, j].imshow(row[key], aspect='auto')
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=13, color='#222222', pad=8)
        axes[i, 0].text(-0.08, 0.5, row_labels[i], transform=axes[i, 0].transAxes,
                         fontsize=11, va='center', ha='right', rotation=90)

    plt.tight_layout()
    path = os.path.join(args.output_dir, 'figureA_robustification_v2.png')
    plt.savefig(path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {path}")
    print("F and C: independent PCA per panel (structurally different signals).")
    print("S / S_r: SHARED PCA basis + normalization range, fit on S. "
          "Any visible difference between these two columns now reflects a "
          "real change relative to S, not independent re-stretching.")


if __name__ == "__main__":
    main()
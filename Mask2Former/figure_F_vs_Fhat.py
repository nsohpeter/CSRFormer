"""
KEY FIGURE: F vs F-hat Across Multiple Domains
=================================================
 
Directly visualizes the central hypothesis across several OOD domains at once,
demonstrating robustness (not a single cherry-picked domain):
 
    Pixel Decoder produces  F     = C + S            (implicit, entangled)
    CSFD produces            F_hat = C + G ⊙ S_r      (explicit, disentangled)
 
Two output modes, controlled by --mode:
 
  --mode teaser   Compact, sized to sit at LaTeX \\columnwidth beside the
                   abstract (HGFormer-style). Drops the redundant "Input"
                   thumbnail column (keeps only F | F_hat) and defaults to
                   a small number of rows so each cell stays legible at
                   final print size. fig-width should be set to your ACTUAL
                   \\columnwidth in inches (not shrunk later by LaTeX) --
                   see the printed guidance at the end of main().
 
  --mode full     The original layout: Input | Mask2Former F | CSFD F_hat,
                   one row per domain, meant for a full-width (figure*)
                   placement later in the paper (e.g. Experiments section).
 
NOTE ON WHAT IS BEING VISUALIZED: the hook captures `mask_features` directly
from `pixel_decoder.forward_features()` -- the raw continuous feature tensor,
BEFORE the transformer decoder, class head, or mask head touch it. This is
NOT a mask prediction. PCA-projected deep features naturally cluster by
semantic region (a well-known phenomenon in feature visualization -- DINO,
MAE, etc.) because that structure is exactly what makes the features usable
for segmentation. What we test is whether that structure survives domain
shift (baseline: collapses) or is preserved (CSFD: retained).
 
PCA basis: fit separately per (domain, representation) pair on that image's
own pixels, then normalized independently. Color mappings are therefore not
literally comparable pixel-for-pixel across different domains, but within
each row (F vs F_hat for the same image) they are directly comparable --
which is the comparison the figure is making.
 
Usage (teaser, for beside the abstract):
    python figure_F_vs_Fhat.py \
        --mode teaser \
        --config custom_configs/training/csfd_swinb_90k.yaml \
        --baseline-weights experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth \
        --csfd-weights experiments/training_outputs/csfd_swinb_90k/model_final.pth \
        --id-dataset cityscapes_fine_sem_seg_val \
        --ood-datasets acdc_night_sem_seg_val gta5_sem_seg_val \
        --image-index 0 \
        --smooth-sigma 1.5 \
        --target-width-in 3.4 \
        --output-dir experiments/figures/key_figure
 
Usage (full, for the Experiments section):
    python figure_F_vs_Fhat.py \
        --mode full \
        --config custom_configs/training/csfd_swinb_90k.yaml \
        --baseline-weights experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth \
        --csfd-weights experiments/training_outputs/csfd_swinb_90k/model_final.pth \
        --id-dataset cityscapes_fine_sem_seg_val \
        --ood-datasets bdd100k_sem_seg_val mapillary_vistas_sem_seg_val gta5_sem_seg_val acdc_night_sem_seg_val \
        --image-index 0 \
        --smooth-sigma 1.5 \
        --output-dir experiments/figures/key_figure
"""
 
import argparse
import os
import sys
 
import numpy as np
import torch
import cv2
from scipy.ndimage import gaussian_filter
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
    """Capture raw mask_features (F) from pixel decoder, before CSFD."""
    pixel_decoder = model.sem_seg_head.pixel_decoder
    orig_fwd = pixel_decoder.forward_features
    def hooked(features):
        mf, ef, ms = orig_fwd(features)
        _FEATURES['F'] = mf.detach()
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
 
 
def get_feature(model, image_tensor, h, w, apply_csfd_module=None):
    """Run forward pass, return F (raw) and optionally F_hat (after CSFD)."""
    _FEATURES.clear()
    with torch.no_grad():
        _ = model([{"image": image_tensor, "height": h, "width": w}])
    F = _FEATURES.get('F')
    if F is None:
        return None, None
    F_hat = None
    if apply_csfd_module is not None:
        with torch.no_grad():
            F_hat = apply_csfd_module(F)
    F_np = F[0].cpu().numpy()
    F_hat_np = F_hat[0].cpu().numpy() if F_hat is not None else None
    return F_np, F_hat_np
 
 
def normalize_rgb(rgb, ref_lo=None, ref_hi=None):
    """Normalize to [0,1]. Shared ref_lo/hi keeps two panels comparable."""
    out = np.zeros_like(rgb)
    lo_out, hi_out = [], []
    for c in range(3):
        if ref_lo is not None:
            lo, hi = ref_lo[c], ref_hi[c]
        else:
            lo, hi = np.percentile(rgb[:, :, c], [1, 99])
        lo_out.append(lo)
        hi_out.append(hi)
        out[:, :, c] = np.clip((rgb[:, :, c] - lo) / (hi - lo + 1e-8), 0, 1)
    return out, lo_out, hi_out
 
 
def smooth_for_display(rgb, sigma=1.5):
    """Mild Gaussian smoothing for DISPLAY ONLY -- does not touch the
    actual features used for prediction, only the visualized RGB."""
    out = np.zeros_like(rgb)
    for c in range(3):
        out[:, :, c] = gaussian_filter(rgb[:, :, c], sigma=sigma)
    return out
 
 
def find_index_by_filename(dataset_name, filename_substr):
    """Find the index of the first image whose file_name contains the
    given substring. Raises if not found."""
    dicts = DatasetCatalog.get(dataset_name)
    for i, d in enumerate(dicts):
        if filename_substr in d['file_name']:
            return i
    raise ValueError(
        f"No image containing '{filename_substr}' found in dataset '{dataset_name}'"
    )
 
 
def load_image_and_features(dataset_name, image_index, transform, device,
                             b_model, c_model, csfd_module):
    """Load one image, return (rgb, F_baseline, F_hat_csfd)."""
    d = DatasetCatalog.get(dataset_name)[image_index]
    bgr = cv2.imread(d['file_name'])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    transformed = transform.get_transform(bgr).apply_image(bgr)
    tensor = torch.as_tensor(
        transformed.astype("float32").transpose(2, 0, 1)).to(device)
    h, w = bgr.shape[0], bgr.shape[1]
 
    F_baseline, _ = get_feature(b_model, tensor, h, w)
    _, F_hat = get_feature(c_model, tensor, h, w, apply_csfd_module=csfd_module)
    return rgb, F_baseline, F_hat
 
 
def pca_project(feat):
    """[C,H,W] -> normalized [H,W,3] RGB via a freshly-fit PCA on this image."""
    C, H, W = feat.shape
    rgb = PCA(n_components=3).fit_transform(feat.reshape(C, -1).T).reshape(H, W, 3)
    rgb_n, _, _ = normalize_rgb(rgb)
    return rgb_n
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["teaser", "full"], default="full",
                    help="'teaser': compact, column-width, F|F_hat only, "
                         "for beside the abstract. 'full': original "
                         "Input|F|F_hat layout for a full-width figure "
                         "later in the paper.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--baseline-weights", required=True)
    ap.add_argument("--csfd-weights", required=True)
    ap.add_argument("--id-dataset", default="cityscapes_fine_sem_seg_val")
    ap.add_argument("--ood-datasets", nargs="+",
                    default=["bdd100k_sem_seg_val", "mapillary_vistas_sem_seg_val",
                             "gta5_sem_seg_val", "acdc_night_sem_seg_val"])
    ap.add_argument("--image-index", type=int, default=0,
                    help="Fallback index used when no filename match given")
    ap.add_argument("--id-filename", type=str, default=None,
                    help="Substring to match a specific source-domain filename "
                         "(overrides --image-index for the source row)")
    ap.add_argument("--ood-filenames", nargs="+", default=None,
                    help="Substrings to match specific filenames, one per "
                         "--ood-datasets entry, in the same order (overrides "
                         "--image-index for those rows)")
    ap.add_argument("--smooth-sigma", type=float, default=1.5,
                    help="Gaussian smoothing sigma for display only (0 = off)")
 
    # --- full-mode sizing (unchanged behavior) ---
    ap.add_argument("--row-height", type=float, default=2.9,
                    help="[full mode] Figure height per row in inches.")
    ap.add_argument("--fig-width", type=float, default=10.5,
                    help="[full mode] Total figure width in inches.")
 
    # --- teaser-mode sizing (NEW) ---
    ap.add_argument("--target-width-in", type=float, default=3.4,
                    help="[teaser mode] Set this to your ACTUAL LaTeX "
                         "\\columnwidth in inches (check your compiled PDF "
                         "or template -- ~3.3-3.5in is typical for a "
                         "two-column US-letter article). The figure is "
                         "rendered AT this physical width so fonts print "
                         "at their true point size with no extra LaTeX "
                         "downscaling.")
    ap.add_argument("--teaser-dpi", type=int, default=400,
                    help="[teaser mode] Higher DPI keeps small text crisp "
                         "since the physical size is already small.")
    ap.add_argument("--include-input", action="store_true", default=True,
                    help="[teaser mode] Include the raw Input photo as a "
                         "third column (default: on).")
    ap.add_argument("--no-include-input", dest="include_input",
                    action="store_false",
                    help="[teaser mode] Drop the Input column, showing "
                         "only F | F_hat (more width per cell, but no "
                         "raw photo for context).")
 
    ap.add_argument("--output-dir", default="experiments/figures/key_figure")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
 
    device = torch.device("cuda")
 
    print("Loading Mask2Former (baseline) model...")
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
    csfd_module = c_model.sem_seg_head.csfd
 
    transform = T.ResizeShortestEdge(
        short_edge_length=b_cfg.INPUT.MIN_SIZE_TEST,
        max_size=b_cfg.INPUT.MAX_SIZE_TEST, sample_style="choice")
 
    # ---- Collect all rows: Source first, then each Target (OOD) domain ----
    id_short = args.id_dataset.replace('_sem_seg_val', '').replace('_fine', '').capitalize()
 
    if args.id_filename:
        id_index = find_index_by_filename(args.id_dataset, args.id_filename)
        print(f"Source: matched '{args.id_filename}' -> index {id_index}")
    else:
        id_index = args.image_index
 
    if args.ood_filenames:
        if len(args.ood_filenames) != len(args.ood_datasets):
            raise ValueError(
                f"--ood-filenames has {len(args.ood_filenames)} entries but "
                f"--ood-datasets has {len(args.ood_datasets)}; must match 1:1"
            )
        ood_indices = []
        for ds, fname in zip(args.ood_datasets, args.ood_filenames):
            idx = find_index_by_filename(ds, fname)
            print(f"Target ({ds}): matched '{fname}' -> index {idx}")
            ood_indices.append(idx)
    else:
        ood_indices = [args.image_index] * len(args.ood_datasets)
 
    all_datasets = [(f"Source\n({id_short})", args.id_dataset, id_index)] + \
                   [(f"Target\n({ds.replace('_sem_seg_val', '').replace('_fine', '')})", ds, idx)
                    for ds, idx in zip(args.ood_datasets, ood_indices)]
 
    if args.mode == "teaser" and len(all_datasets) > 3:
        print(f"[teaser mode] {len(all_datasets)} rows requested -- teaser "
              f"figures stay legible at column width with ~2-3 rows max. "
              f"Consider trimming --ood-datasets to your 1-2 most striking "
              f"cases (e.g. acdc_night) and use --mode full for the "
              f"complete comparison elsewhere in the paper.")
 
    rows = []
    for label, ds, idx in all_datasets:
        print(f"Processing {label.strip()} ({ds}, index {idx})...")
        img, F_raw, Fhat_raw = load_image_and_features(
            ds, idx, transform, device, b_model, c_model, csfd_module)
        rows.append({'label': label, 'img': img, 'F': F_raw, 'Fhat': Fhat_raw})
 
    n_rows = len(rows)
 
    if args.mode == "full":
        # ---- Original layout: Input | F | F_hat ----
        n_cols = 3
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(args.fig_width, args.row_height * n_rows),
            gridspec_kw={'wspace': 0.02, 'hspace': 0.04})
        if n_rows == 1:
            axes = axes[np.newaxis, :]
 
        for i, row in enumerate(rows):
            rgb_F = pca_project(row['F'])
            rgb_Fhat = pca_project(row['Fhat'])
            if args.smooth_sigma > 0:
                rgb_F = smooth_for_display(rgb_F, args.smooth_sigma)
                rgb_Fhat = smooth_for_display(rgb_Fhat, args.smooth_sigma)
 
            axes[i, 0].imshow(row['img']); axes[i, 0].axis('off')
            axes[i, 1].imshow(rgb_F);      axes[i, 1].axis('off')
            axes[i, 2].imshow(rgb_Fhat);   axes[i, 2].axis('off')
 
            axes[i, 0].text(-0.12, 0.5, row['label'], transform=axes[i, 0].transAxes,
                             fontsize=9, fontweight='bold', ha='right', va='center',
                             linespacing=1.4)
 
            if i == 0:
                axes[i, 0].set_title('Input', fontsize=11, fontweight='bold', pad=4)
                axes[i, 1].set_title(r'Mask2Former $F$', fontsize=11,
                                      fontweight='bold', color='#333333', pad=4)
                axes[i, 2].set_title(r'CSFD $\hat{F}$ (ours)', fontsize=11,
                                      fontweight='bold', color='#006600', pad=4)
 
        fig.suptitle(
            r'$F=C+S$ (entangled)  vs.  $\hat{F}=C+G\odot S_r$ (disentangled)',
            fontsize=11, fontweight='bold', y=1.0)
        plt.tight_layout(rect=[0.08, 0, 1, 0.99])
 
        dpi = 200
        suffix = "full"
 
    else:
        # ---- Teaser layout, sized AT final print width ----
        # Row height is DERIVED from the real image aspect ratio, not
        # guessed -- otherwise imshow letterboxes each cell (pads with
        # white top/bottom to preserve aspect ratio) and the whole figure
        # ends up looking like it has huge gaps between rows, when it's
        # actually blank padding *inside* each axes box.
        include_input = args.include_input
        n_cols = 3 if include_input else 2
 
        # Use the first row's actual image to measure aspect ratio (W/H).
        # Feature maps and the input photo share the same aspect ratio
        # (features are just a uniformly-strided version of the input).
        sample_h, sample_w = rows[0]['img'].shape[:2]
        aspect_w_over_h = sample_w / sample_h
 
        wspace_frac = 0.025
        hspace_frac = 0.05
        axes_width_in = (args.target_width_in
                          * (1 - wspace_frac * (n_cols - 1)) / n_cols)
        axes_height_in = axes_width_in / aspect_w_over_h
        title_pad_in = 0.22   # rough allowance for the column-title row
        fig_height_in = (axes_height_in * n_rows
                          * (1 + hspace_frac)) + title_pad_in
 
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(args.target_width_in, fig_height_in),
            gridspec_kw={'wspace': wspace_frac, 'hspace': hspace_frac})
        if n_rows == 1:
            axes = axes[np.newaxis, :]
 
        for i, row in enumerate(rows):
            rgb_F = pca_project(row['F'])
            rgb_Fhat = pca_project(row['Fhat'])
            if args.smooth_sigma > 0:
                rgb_F = smooth_for_display(rgb_F, args.smooth_sigma)
                rgb_Fhat = smooth_for_display(rgb_Fhat, args.smooth_sigma)
 
            col = 0
            if include_input:
                axes[i, col].imshow(row['img']); axes[i, col].axis('off')
                # Row label to the left of the Input panel, like full mode.
                axes[i, col].text(-0.10, 0.5, row['label'],
                                   transform=axes[i, col].transAxes,
                                   fontsize=7, fontweight='bold',
                                   ha='right', va='center', linespacing=1.3)
                col += 1
            axes[i, col].imshow(rgb_F); axes[i, col].axis('off')
            if not include_input:
                # No Input column to anchor the label to -- overlay it.
                axes[i, col].text(0.03, 0.94, row['label'].replace('\n', ' '),
                                   transform=axes[i, col].transAxes,
                                   fontsize=7, fontweight='bold', color='white',
                                   ha='left', va='top',
                                   bbox=dict(boxstyle='round,pad=0.2',
                                             facecolor='black', alpha=0.55,
                                             edgecolor='none'))
            col += 1
            axes[i, col].imshow(rgb_Fhat); axes[i, col].axis('off')
 
            if i == 0:
                # Kept terse on purpose: at true column width each cell is
                # only ~1in wide, so "Mask2Former F" / "CSFD F_hat (ours)"
                # collide with their neighbors. Full names belong in the
                # LaTeX \caption, not baked into the raster -- see the
                # suggested caption text printed at the end of this script.
                c0 = 1 if include_input else 0
                if include_input:
                    axes[i, 0].set_title('Input', fontsize=8,
                                          fontweight='bold', pad=3)
                axes[i, c0].set_title(r'$F$', fontsize=9,
                                       fontweight='bold', color='#333333', pad=3)
                axes[i, c0 + 1].set_title(r'$\hat{F}$ (ours)', fontsize=9,
                                           fontweight='bold', color='#006600', pad=3)
 
        plt.tight_layout(pad=0.25)
 
        dpi = args.teaser_dpi
        suffix = "teaser"
 
    path = os.path.join(
        args.output_dir, f'F_vs_Fhat_multidomain_{suffix}_idx{args.image_index}.png')
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {path}")
 
    if args.mode == "teaser":
        print(
            f"\nThis PNG was rendered at {args.target_width_in}in wide -- "
            f"if that matches your real \\columnwidth, use it in LaTeX with "
            f"\\includegraphics[width=\\columnwidth]{{...}} and it will need "
            f"little to no rescaling, so the fonts above will print at "
            f"roughly their true point size. If your actual \\columnwidth "
            f"differs, re-run with --target-width-in set to that value "
            f"rather than rescaling this PNG afterward.\n"
            f"\n"
            f"Column headers were kept to bare '$F$' / '$\\hat{{F}}$' since "
            f"there's no room for full names at this width -- put the full "
            f"names in your LaTeX \\caption, e.g.:\n"
            f'  \\caption{{Mask2Former features $F$ (left) vs.\\ our '
            f"CSFD-reconstructed features $\\hat{{F}}$ (right) across "
            f"source and unseen target domains.}}"
        )
 
 
if __name__ == "__main__":
    main()





# """
# KEY FIGURE: F vs F-hat Across Multiple Domains
# =================================================
 
# Directly visualizes the central hypothesis across several OOD domains at once,
# demonstrating robustness (not a single cherry-picked domain):
 
#     Pixel Decoder produces  F     = C + S            (implicit, entangled)
#     CSFD produces            F_hat = C + G ⊙ S_r      (explicit, disentangled)
 
# Layout: compact grid, one row per domain (ID first, then each OOD domain),
# 3 columns: Input | Mask2Former F | CSFD F_hat.
 
# NOTE ON WHAT IS BEING VISUALIZED: the hook captures `mask_features` directly
# from `pixel_decoder.forward_features()` — the raw continuous feature tensor,
# BEFORE the transformer decoder, class head, or mask head touch it. This is
# NOT a mask prediction. PCA-projected deep features naturally cluster by
# semantic region (a well-known phenomenon in feature visualization — DINO,
# MAE, etc.) because that structure is exactly what makes the features usable
# for segmentation. What we test is whether that structure survives domain
# shift (baseline: collapses) or is preserved (CSFD: retained).
 
# PCA basis: fit separately per (domain, representation) pair on that image's
# own pixels, then normalized independently. Color mappings are therefore not
# literally comparable pixel-for-pixel across different domains, but within
# each row (F vs F_hat for the same image) they are directly comparable —
# which is the comparison the figure is making.
 
# Usage:
#     python figure_F_vs_Fhat.py \
#         --config custom_configs/training/csfd_swinb_90k.yaml \
#         --baseline-weights experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth \
#         --csfd-weights experiments/training_outputs/csfd_swinb_90k/model_final.pth \
#         --id-dataset cityscapes_fine_sem_seg_val \
#         --ood-datasets bdd100k_sem_seg_val mapillary_vistas_sem_seg_val gta5_sem_seg_val acdc_night_sem_seg_val \
#         --image-index 0 \
#         --smooth-sigma 1.5 \
#         --output-dir experiments/figures/key_figure
# """
 
# import argparse
# import os
# import sys
 
# import numpy as np
# import torch
# import cv2
# from scipy.ndimage import gaussian_filter
# from sklearn.decomposition import PCA
 
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
#     """Capture raw mask_features (F) from pixel decoder, before CSFD."""
#     pixel_decoder = model.sem_seg_head.pixel_decoder
#     orig_fwd = pixel_decoder.forward_features
#     def hooked(features):
#         mf, ef, ms = orig_fwd(features)
#         _FEATURES['F'] = mf.detach()
#         return mf, ef, ms
#     pixel_decoder.forward_features = hooked
 
 
# def build_cfg(config_file, weights, csfd_enabled):
#     cfg = get_cfg()
#     add_deeplab_config(cfg)
#     add_maskformer2_config(cfg)
#     cfg.merge_from_file(config_file)
#     cfg.defrost()
#     cfg.MODEL.WEIGHTS = weights
#     cfg.MODEL.CSFD.ENABLED = csfd_enabled
#     cfg.freeze()
#     return cfg
 
 
# def get_feature(model, image_tensor, h, w, apply_csfd_module=None):
#     """Run forward pass, return F (raw) and optionally F_hat (after CSFD)."""
#     _FEATURES.clear()
#     with torch.no_grad():
#         _ = model([{"image": image_tensor, "height": h, "width": w}])
#     F = _FEATURES.get('F')
#     if F is None:
#         return None, None
#     F_hat = None
#     if apply_csfd_module is not None:
#         with torch.no_grad():
#             F_hat = apply_csfd_module(F)
#     F_np = F[0].cpu().numpy()
#     F_hat_np = F_hat[0].cpu().numpy() if F_hat is not None else None
#     return F_np, F_hat_np
 
 
# def normalize_rgb(rgb, ref_lo=None, ref_hi=None):
#     """Normalize to [0,1]. Shared ref_lo/hi keeps two panels comparable."""
#     out = np.zeros_like(rgb)
#     lo_out, hi_out = [], []
#     for c in range(3):
#         if ref_lo is not None:
#             lo, hi = ref_lo[c], ref_hi[c]
#         else:
#             lo, hi = np.percentile(rgb[:, :, c], [1, 99])
#         lo_out.append(lo)
#         hi_out.append(hi)
#         out[:, :, c] = np.clip((rgb[:, :, c] - lo) / (hi - lo + 1e-8), 0, 1)
#     return out, lo_out, hi_out
 
 
# def smooth_for_display(rgb, sigma=1.5):
#     """Mild Gaussian smoothing for DISPLAY ONLY — does not touch the
#     actual features used for prediction, only the visualized RGB."""
#     out = np.zeros_like(rgb)
#     for c in range(3):
#         out[:, :, c] = gaussian_filter(rgb[:, :, c], sigma=sigma)
#     return out
 
 
# def find_index_by_filename(dataset_name, filename_substr):
#     """Find the index of the first image whose file_name contains the
#     given substring. Raises if not found."""
#     dicts = DatasetCatalog.get(dataset_name)
#     for i, d in enumerate(dicts):
#         if filename_substr in d['file_name']:
#             return i
#     raise ValueError(
#         f"No image containing '{filename_substr}' found in dataset '{dataset_name}'"
#     )
 
 
# def load_image_and_features(dataset_name, image_index, transform, device,
#                              b_model, c_model, csfd_module):
#     """Load one image, return (rgb, F_baseline, F_hat_csfd)."""
#     d = DatasetCatalog.get(dataset_name)[image_index]
#     bgr = cv2.imread(d['file_name'])
#     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#     transformed = transform.get_transform(bgr).apply_image(bgr)
#     tensor = torch.as_tensor(
#         transformed.astype("float32").transpose(2, 0, 1)).to(device)
#     h, w = bgr.shape[0], bgr.shape[1]
 
#     F_baseline, _ = get_feature(b_model, tensor, h, w)
#     _, F_hat = get_feature(c_model, tensor, h, w, apply_csfd_module=csfd_module)
#     return rgb, F_baseline, F_hat
 
 
# def pca_project(feat):
#     """[C,H,W] -> normalized [H,W,3] RGB via a freshly-fit PCA on this image."""
#     C, H, W = feat.shape
#     rgb = PCA(n_components=3).fit_transform(feat.reshape(C, -1).T).reshape(H, W, 3)
#     rgb_n, _, _ = normalize_rgb(rgb)
#     return rgb_n
 
 
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", required=True)
#     ap.add_argument("--baseline-weights", required=True)
#     ap.add_argument("--csfd-weights", required=True)
#     ap.add_argument("--id-dataset", default="cityscapes_fine_sem_seg_val")
#     ap.add_argument("--ood-datasets", nargs="+",
#                     default=["bdd100k_sem_seg_val", "mapillary_vistas_sem_seg_val",
#                              "gta5_sem_seg_val", "acdc_night_sem_seg_val"])
#     ap.add_argument("--image-index", type=int, default=0,
#                     help="Fallback index used when no filename match given")
#     ap.add_argument("--id-filename", type=str, default=None,
#                     help="Substring to match a specific source-domain filename "
#                          "(overrides --image-index for the source row)")
#     ap.add_argument("--ood-filenames", nargs="+", default=None,
#                     help="Substrings to match specific filenames, one per "
#                          "--ood-datasets entry, in the same order (overrides "
#                          "--image-index for those rows)")
#     ap.add_argument("--smooth-sigma", type=float, default=1.5,
#                     help="Gaussian smoothing sigma for display only (0 = off)")
#     ap.add_argument("--row-height", type=float, default=2.9,
#                     help="Figure height per row in inches (default 2.9). "
#                          "Use ~1.6-1.8 for a compact conference-paper figure.")
#     ap.add_argument("--fig-width", type=float, default=10.5,
#                     help="Total figure width in inches (default 10.5)")
#     ap.add_argument("--output-dir", default="experiments/figures/key_figure")
#     args = ap.parse_args()
#     os.makedirs(args.output_dir, exist_ok=True)
 
#     device = torch.device("cuda")
 
#     print("Loading Mask2Former (baseline) model...")
#     b_cfg = build_cfg(args.config, args.baseline_weights, csfd_enabled=False)
#     b_model = build_model(b_cfg)
#     b_model.eval()
#     DetectionCheckpointer(b_model).load(args.baseline_weights)
#     register_feature_hook(b_model)
 
#     print("Loading CSFD model...")
#     c_cfg = build_cfg(args.config, args.csfd_weights, csfd_enabled=True)
#     c_model = build_model(c_cfg)
#     c_model.eval()
#     DetectionCheckpointer(c_model).load(args.csfd_weights)
#     register_feature_hook(c_model)
#     csfd_module = c_model.sem_seg_head.csfd
 
#     transform = T.ResizeShortestEdge(
#         short_edge_length=b_cfg.INPUT.MIN_SIZE_TEST,
#         max_size=b_cfg.INPUT.MAX_SIZE_TEST, sample_style="choice")
 
#     # ---- Collect all rows: Source first, then each Target (OOD) domain ----
#     id_short = args.id_dataset.replace('_sem_seg_val', '').replace('_fine', '').capitalize()
 
#     # Resolve the index for the source row
#     if args.id_filename:
#         id_index = find_index_by_filename(args.id_dataset, args.id_filename)
#         print(f"Source: matched '{args.id_filename}' -> index {id_index}")
#     else:
#         id_index = args.image_index
 
#     # Resolve indices for each OOD row
#     if args.ood_filenames:
#         if len(args.ood_filenames) != len(args.ood_datasets):
#             raise ValueError(
#                 f"--ood-filenames has {len(args.ood_filenames)} entries but "
#                 f"--ood-datasets has {len(args.ood_datasets)}; must match 1:1"
#             )
#         ood_indices = []
#         for ds, fname in zip(args.ood_datasets, args.ood_filenames):
#             idx = find_index_by_filename(ds, fname)
#             print(f"Target ({ds}): matched '{fname}' -> index {idx}")
#             ood_indices.append(idx)
#     else:
#         ood_indices = [args.image_index] * len(args.ood_datasets)
 
#     all_datasets = [(f"Source\n({id_short})", args.id_dataset, id_index)] + \
#                    [(f"Target\n({ds.replace('_sem_seg_val', '').replace('_fine', '')})", ds, idx)
#                     for ds, idx in zip(args.ood_datasets, ood_indices)]
 
#     rows = []
#     for label, ds, idx in all_datasets:
#         print(f"Processing {label.strip()} ({ds}, index {idx})...")
#         img, F_raw, Fhat_raw = load_image_and_features(
#             ds, idx, transform, device, b_model, c_model, csfd_module)
#         rows.append({'label': label, 'img': img, 'F': F_raw, 'Fhat': Fhat_raw})
 
#     # ---- Build compact grid: N rows x 3 columns ----
#     n_rows = len(rows)
#     fig, axes = plt.subplots(n_rows, 3, figsize=(args.fig_width, args.row_height * n_rows),
#                               gridspec_kw={'wspace': 0.02, 'hspace': 0.04})
#     if n_rows == 1:
#         axes = axes[np.newaxis, :]
 
#     for i, row in enumerate(rows):
#         rgb_F = pca_project(row['F'])
#         rgb_Fhat = pca_project(row['Fhat'])
#         if args.smooth_sigma > 0:
#             rgb_F = smooth_for_display(rgb_F, args.smooth_sigma)
#             rgb_Fhat = smooth_for_display(rgb_Fhat, args.smooth_sigma)
 
#         axes[i, 0].imshow(row['img'])
#         axes[i, 0].axis('off')
#         axes[i, 1].imshow(rgb_F)
#         axes[i, 1].axis('off')
#         axes[i, 2].imshow(rgb_Fhat)
#         axes[i, 2].axis('off')
 
#         # Row label on the far left, outside the axes
#         axes[i, 0].text(-0.12, 0.5, row['label'], transform=axes[i, 0].transAxes,
#                          fontsize=9, fontweight='bold', ha='right', va='center',
#                          linespacing=1.4)
 
#         if i == 0:
#             axes[i, 0].set_title('Input', fontsize=11, fontweight='bold', pad=4)
#             axes[i, 1].set_title(r'Mask2Former $F$', fontsize=11,
#                                   fontweight='bold', color='#333333', pad=4)
#             axes[i, 2].set_title(r'CSFD $\hat{F}$ (ours)', fontsize=11,
#                                   fontweight='bold', color='#006600', pad=4)
 
#     fig.suptitle(
#         r'$F=C+S$ (entangled)  vs.  $\hat{F}=C+G\odot S_r$ (disentangled)',
#         fontsize=11, fontweight='bold', y=1.0)
#     plt.tight_layout(rect=[0.08, 0, 1, 0.99])
 
#     path = os.path.join(args.output_dir, f'F_vs_Fhat_multidomain_idx{args.image_index}.png')
#     plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
#     plt.close()
#     print(f"\nSaved: {path}")
 
 
# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
STEP 3 (v2) — Assemble the 4-row qualitative figure for the paper.

Rebuilt for the specific row/story spec:
    1. Cityscapes — source domain, Ours ~= GT, baselines show pole/person errors
    2. Mapillary  — Ours matches GT, baselines mislabel car / confuse vegetation
    3. GTA5       — Ours clean, baselines show sky noise + road-boundary drift
    4. Night (ACDC) — the honest weakness: Ours is the noisiest here, baselines
       are cleaner. This row is KEPT IN because it matches the actual results
       table (CSR does not beat baseline on Night) -- we do not cherry-pick
       only winning domains.

Sizing spec (explicit, from the user):
    - overall figure size: 7.0 x 4.4 in   -> fits \\textwidth in a two-column
      paper (this is a \\textwidth figure, NOT a single-column one like Fig1).
    - panel aspect ratio: 2:1 (width:height), matching typical driving-scene
      photos (Cityscapes/GTA5/Mapillary/ACDC are all close to 2:1 already).
    - no automatic error-box overlay. Yellow boxes are added manually by the
      user in post (e.g. in PowerPoint/Illustrator on top of the exported
      PNG/PDF) -- do not draw any boxes here.

Pure numpy/matplotlib, no detectron2/mask2former imports. Run once Step 1
(infer_m2f_csr.py) and Step 2 (infer_cmformer.py) have both saved .npy files
for all 4 rows into the same --input-dir.

Usage:
    python assemble_figure_v2.py \\
        --input-dir experiments/figures/qualitative/raw \\
        --labels Cityscapes Mapillary GTA5 Night \\
        --output-dir experiments/figures/qualitative
"""

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32],
], dtype=np.uint8)

# Fixed layout spec (do not derive from per-image aspect -- all four source
# domains are already close to 2:1, and the user wants a uniform, clean grid
# rather than a per-row-aspect-driven one like the older 5-domain version).
FIG_WIDTH_IN = 7.0
FIG_HEIGHT_IN = 4.4
PANEL_ASPECT_H_OVER_W = 0.5  # 2:1 width:height -> height/width = 0.5

N_COLS = 5
COL_TITLES = ['Image', 'Ground Truth', 'Mask2Former', 'CMFormer', r'Ours']


def colorize(pred, palette=CITYSCAPES_PALETTE):
    h, w = pred.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(len(palette)):
        color[pred == cls_id] = palette[cls_id]
    return color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--labels", nargs="+",
                    default=["Cityscapes", "Mapillary", "GTA5", "Night"],
                    help="Row labels matching the keys saved by Steps 1/2, "
                         "in display order top-to-bottom.")
    ap.add_argument("--output-dir", default="experiments/figures/qualitative")
    ap.add_argument("--out-name", default="qualitative_predictions_4row")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for label in args.labels:
        key = label.lower().replace(' ', '_')
        rgb = np.load(os.path.join(args.input_dir, f"{key}_rgb.npy"))
        gt_path = os.path.join(args.input_dir, f"{key}_gt.npy")
        gt = np.load(gt_path) if os.path.exists(gt_path) else None
        m2f = np.load(os.path.join(args.input_dir, f"{key}_m2f.npy"))
        cmf = np.load(os.path.join(args.input_dir, f"{key}_cmf.npy"))
        csr = np.load(os.path.join(args.input_dir, f"{key}_csr.npy"))
        rows.append({'label': label, 'rgb': rgb, 'gt': gt,
                     'm2f': m2f, 'cmf': cmf, 'csr': csr})
        print(f"Loaded {label}")

    n = len(rows)

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    # Uniform height ratios (every row gets the same box) -- the 2:1 panel
    # aspect is enforced per-axes via set_box_aspect below, not through
    # height_ratios, so this grid stays a plain uniform n x 5 layout.
    gs = fig.add_gridspec(
        n, N_COLS,
        hspace=0.10, wspace=0.02,
        left=0.01, right=0.99, top=0.93, bottom=0.01,
    )

    for i, row in enumerate(rows):
        panels = [row['rgb'],
                  colorize(row['gt']) if row['gt'] is not None else None,
                  colorize(row['m2f']),
                  colorize(row['cmf']),
                  colorize(row['csr'])]

        for j, img in enumerate(panels):
            ax = fig.add_subplot(gs[i, j])
            if img is not None:
                # aspect='auto' -- fixes the imshow letterboxing/row-gap bug
                # from the earlier Figure A / Figure B scripts: 'equal' (the
                # default) pads each panel to preserve pixel aspect inside a
                # mismatched axes box, which is what caused the big vertical
                # gaps there. 'auto' lets the panel fill its 2:1 gridspec box
                # exactly, which is also what we want visually here since all
                # four source photos are already close to 2:1 themselves.
                ax.imshow(img, aspect='auto')
            ax.axis('off')
            ax.set_box_aspect(PANEL_ASPECT_H_OVER_W)

            if i == 0:
                ax.set_title(COL_TITLES[j], fontsize=9, pad=4)

        # Row label, bottom-left of the input image panel (HGFormer style).
        # NOTE: no error boxes are drawn here -- those are added manually by
        # the user afterward, directly on the exported PNG/PDF.
        img_ax = fig.axes[i * N_COLS]
        img_ax.text(0.03, 0.08, row['label'], transform=img_ax.transAxes,
                    fontsize=8, fontweight='bold', color='yellow',
                    ha='left', va='bottom',
                    bbox=dict(facecolor='black', alpha=0.45, pad=1.5, edgecolor='none'))

    for ext in ('png', 'pdf'):
        path = os.path.join(args.output_dir, f"{args.out_name}.{ext}")
        plt.savefig(path, dpi=400 if ext == 'png' else None,
                    bbox_inches='tight', pad_inches=0.03, facecolor='white')
        print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
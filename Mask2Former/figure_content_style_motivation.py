"""
Motivational Figure: Content is Invariant, Style Varies
==========================================================
 
Demonstrates the core hypothesis behind CSFD using real images from four
domains: the SAME semantic content (road, car, building, vegetation) is
present everywhere, but visual STYLE (weather, lighting, rendering, camera)
varies drastically.
 
Design: fan-out layout (different from side-by-side comparison figures).
    - Top: one unified "shared content" banner (green chips)
    - Middle: connecting lines fanning down to each domain
    - Bottom: 4 real photos, one per domain, each labeled with its own
      style attributes in orange — visually showing one shared truth
      branching into many different appearances.
 
Usage:
    python figure_content_style_motivation.py \
        --output-dir experiments/figures/motivation
"""
 
import argparse
import os
import sys
import random
 
import cv2
import numpy as np
 
sys.path.insert(0, os.path.abspath("."))
from mask2former import add_maskformer2_config
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.projects.deeplab import add_deeplab_config
 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
 
 
# Domain configs: (dataset_name, display_name, style_description, image_filename_hint)
DOMAIN_CONFIGS = [
    ("cityscapes_fine_sem_seg_val", "Cityscapes",
     "clear, daytime,\nEuropean street", None),
    ("bdd100k_sem_seg_val", "BDD100K",
     "overcast, daytime,\nAmerican street", None),
    ("gta5_sem_seg_val", "GTA5",
     "synthetic, sunny,\nrendered scene", None),
    ("acdc_night_sem_seg_val", "ACDC-Night",
     "dark, nighttime,\nartificial lighting", None),
]
 
CONTENT_CHIPS = ["road", "car", "building", "vegetation", "sky", "pedestrian"]
 
 
def find_index_by_filename(dataset_name, filename_substr):
    dicts = DatasetCatalog.get(dataset_name)
    for i, d in enumerate(dicts):
        if filename_substr in d['file_name']:
            return i
    return None
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-index", type=int, default=0,
                    help="Fallback index for each domain (default 0)")
    ap.add_argument("--cityscapes-filename", type=str, default=None)
    ap.add_argument("--bdd-filename", type=str, default=None)
    ap.add_argument("--gta5-filename", type=str, default=None)
    ap.add_argument("--acdc-night-filename", type=str, default=None)
    ap.add_argument("--output-dir", default="experiments/figures/motivation")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
 
    filename_overrides = {
        "cityscapes_fine_sem_seg_val": args.cityscapes_filename,
        "bdd100k_sem_seg_val": args.bdd_filename,
        "gta5_sem_seg_val": args.gta5_filename,
        "acdc_night_sem_seg_val": args.acdc_night_filename,
    }
 
    # ---- Load one real image per domain ----
    images = []
    for ds_name, display_name, style_desc, _ in DOMAIN_CONFIGS:
        override = filename_overrides.get(ds_name)
        if override:
            idx = find_index_by_filename(ds_name, override)
            if idx is None:
                print(f"WARNING: '{override}' not found in {ds_name}, using index 0")
                idx = args.image_index
        else:
            idx = args.image_index
 
        d = DatasetCatalog.get(ds_name)[idx]
        bgr = cv2.imread(d['file_name'])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        images.append(rgb)
        print(f"{display_name}: loaded {d['file_name']}")
 
    # ---- Build the fan-out figure ----
    n = len(DOMAIN_CONFIGS)
    fig = plt.figure(figsize=(3.2 * n, 5.2))
 
    # Grid: 1 row for the content banner (spans full width),
    #       1 row for connector lines (invisible, just spacing),
    #       1 row for the 4 images
    gs = fig.add_gridspec(3, n, height_ratios=[0.55, 0.18, 2.4], hspace=0.08, wspace=0.06)
 
    # ---- Row 0: shared content banner (single wide axis) ----
    ax_banner = fig.add_subplot(gs[0, :])
    ax_banner.set_xlim(0, 1)
    ax_banner.set_ylim(0, 1)
    ax_banner.axis('off')
 
    ax_banner.text(0.5, 0.92, 'Shared Semantic Content (domain-invariant)',
                    ha='center', va='top', fontsize=13, fontweight='bold',
                    color='#1a6b1a')
 
    n_chips = len(CONTENT_CHIPS)
    chip_w = 0.9 / n_chips
    start_x = 0.05
    for i, chip in enumerate(CONTENT_CHIPS):
        cx = start_x + i * chip_w + chip_w / 2
        box = FancyBboxPatch(
            (cx - chip_w * 0.42, 0.15), chip_w * 0.84, 0.45,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor='#e8f7e8', edgecolor='#1a6b1a', linewidth=1.3,
            transform=ax_banner.transData
        )
        ax_banner.add_patch(box)
        ax_banner.text(cx, 0.375, chip, ha='center', va='center',
                        fontsize=10.5, fontweight='bold', color='#1a6b1a')
 
    # ---- Row 1: connector lines fanning down from banner to each image ----
    ax_connect = fig.add_subplot(gs[1, :])
    ax_connect.set_xlim(0, 1)
    ax_connect.set_ylim(0, 1)
    ax_connect.axis('off')
 
    for i in range(n):
        cx = (i + 0.5) / n
        ax_connect.annotate('', xy=(cx, 0.05), xytext=(cx, 0.95),
                             arrowprops=dict(arrowstyle='-|>', color='#1a6b1a',
                                              lw=1.6, alpha=0.75,
                                              connectionstyle="arc3,rad=0"))
 
    # ---- Row 2: the four real domain images ----
    for i, ((ds_name, display_name, style_desc, _), img) in enumerate(
            zip(DOMAIN_CONFIGS, images)):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(img)
        ax.axis('off')
 
        # Domain name above the image (as an axis title)
        ax.set_title(display_name, fontsize=12, fontweight='bold',
                     color='#333333', pad=6)
 
        # Style descriptor below the image, in orange, framed
        ax.text(0.5, -0.10, style_desc, transform=ax.transAxes,
                ha='center', va='top', fontsize=9.5, color='#b35c00',
                fontweight='bold', linespacing=1.3,
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#fff3e0',
                          edgecolor='#b35c00', linewidth=1.0))
 
    fig.suptitle(
        'Same Content, Different Style: The Foundation of Domain Generalization',
        fontsize=14, fontweight='bold', y=1.0)
 
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
 
    path = os.path.join(args.output_dir, 'content_style_motivation.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {path}")
 
 
if __name__ == "__main__":
    main()
"""
Qualitative Prediction Comparison — HGFormer-style clean grid
================================================================
Columns: Input | Ground Truth | Mask2Former | CMFormer (ours) | CSR+SA (ours)
Rows: one per ACDC condition (default: Night, Snow — the two strongest deltas)
 
Matches HGFormer Figure 4's clean convention: no report-style headers, no
error overlays, just colorized predictions side by side with a caption below.
 
Usage:
    python figure_qualitative_predictions.py \
        --config custom_configs/training/csfd_swinb_90k.yaml \
        --m2f-weights experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth \
        --cmformer-config <path to CMFormer repo's config, if different codebase> \
        --cmformer-weights <path to CMFormer checkpoint> \
        --csr-weights experiments/training_outputs/csfd_styleaug_targeted_swinb_90k/model_final.pth \
        --conditions night snow \
        --image-filenames GOPR0351_frame_000651_rgb_anon <snow_filename> \
        --output-dir experiments/figures/qualitative
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
 
# Cityscapes 19-class palette (matches HGFormer/CMFormer figure conventions)
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],   # traffic light
    [220, 220, 0],    # traffic sign
    [107, 142, 35],   # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],   # sky
    [220, 20, 60],    # person
    [255, 0, 0],      # rider
    [0, 0, 142],      # car
    [0, 0, 70],       # truck
    [0, 60, 100],     # bus
    [0, 80, 100],     # train
    [0, 0, 230],      # motorcycle
    [119, 11, 32],    # bicycle
], dtype=np.uint8)
IGNORE_LABEL = 255
 
 
def colorize(pred, palette=CITYSCAPES_PALETTE):
    h, w = pred.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(len(palette)):
        color[pred == cls_id] = palette[cls_id]
    return color
 
 
def find_index_by_filename(dataset_name, filename_substr):
    dicts = DatasetCatalog.get(dataset_name)
    for i, d in enumerate(dicts):
        if filename_substr in d['file_name']:
            return i
    return None
 
 
def build_cfg_generic(config_file, weights, csfd_enabled=None):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = weights
    if csfd_enabled is not None and hasattr(cfg.MODEL, 'CSFD'):
        cfg.MODEL.CSFD.ENABLED = csfd_enabled
    cfg.freeze()
    return cfg
 
 
def get_prediction(model, image_tensor, h, w):
    with torch.no_grad():
        outputs = model([{"image": image_tensor, "height": h, "width": w}])
    return outputs[0]["sem_seg"].argmax(dim=0).cpu().numpy()
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Config for Mask2Former/CSR (same codebase, CSFD flag toggled)")
    ap.add_argument("--m2f-weights", required=True)
    ap.add_argument("--cmformer-config", required=True,
                    help="CMFormer's own config file (different decoder, may be different repo)")
    ap.add_argument("--cmformer-weights", required=True)
    ap.add_argument("--csr-weights", required=True,
                    help="CSR (+ style-aug targeted) checkpoint, same config as --config")
    ap.add_argument("--conditions", nargs="+", default=["night", "snow"])
    ap.add_argument("--image-filenames", nargs="+", default=None,
                    help="Optional filename substring per condition, same order as --conditions")
    ap.add_argument("--output-dir", default="experiments/figures/qualitative")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
 
    device = torch.device("cuda")
 
    # ---- Build all three models ----
    print("Loading Mask2Former...")
    m2f_cfg = build_cfg_generic(args.config, args.m2f_weights, csfd_enabled=False)
    m2f_model = build_model(m2f_cfg)
    m2f_model.eval()
    DetectionCheckpointer(m2f_model).load(args.m2f_weights)
 
    print("Loading CMFormer...")
    cmf_cfg = build_cfg_generic(args.cmformer_config, args.cmformer_weights, csfd_enabled=None)
    cmf_model = build_model(cmf_cfg)
    cmf_model.eval()
    DetectionCheckpointer(cmf_model).load(args.cmformer_weights)
 
    print("Loading CSR (+StyleAug targeted)...")
    csr_cfg = build_cfg_generic(args.config, args.csr_weights, csfd_enabled=True)
    csr_model = build_model(csr_cfg)
    csr_model.eval()
    DetectionCheckpointer(csr_model).load(args.csr_weights)
 
    transform = T.ResizeShortestEdge(
        short_edge_length=m2f_cfg.INPUT.MIN_SIZE_TEST,
        max_size=m2f_cfg.INPUT.MAX_SIZE_TEST, sample_style="choice")
 
    # ---- Collect rows ----
    rows = []
    for i, cond in enumerate(args.conditions):
        ds_name = f"acdc_{cond}_sem_seg_val"
        print(f"\nCondition: {cond}")
 
        if args.image_filenames and i < len(args.image_filenames):
            idx = find_index_by_filename(ds_name, args.image_filenames[i])
            if idx is None:
                print(f"  filename not found, using index 0")
                idx = 0
        else:
            idx = 0
 
        d = DatasetCatalog.get(ds_name)[idx]
        bgr = cv2.imread(d['file_name'])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = bgr.shape[0], bgr.shape[1]
 
        gt = None
        if d.get('sem_seg_file_name'):
            gt = cv2.imread(d['sem_seg_file_name'], cv2.IMREAD_UNCHANGED)
 
        transformed = transform.get_transform(bgr).apply_image(bgr)
        tensor = torch.as_tensor(
            transformed.astype("float32").transpose(2, 0, 1)).to(device)
 
        print("  Mask2Former inference...")
        m2f_pred = get_prediction(m2f_model, tensor, h, w)
 
        print("  CMFormer inference...")
        cmf_pred = get_prediction(cmf_model, tensor, h, w)
 
        print("  CSR inference...")
        csr_pred = get_prediction(csr_model, tensor, h, w)
 
        rows.append({
            'label': cond.capitalize(),
            'rgb': rgb,
            'gt': gt,
            'm2f': m2f_pred,
            'cmf': cmf_pred,
            'csr': csr_pred,
        })
 
    # ---- Build figure: HGFormer-style clean grid ----
    n = len(rows)
    fig, axes = plt.subplots(n, 5, figsize=(15, 3.0 * n),
                              gridspec_kw={'wspace': 0.02, 'hspace': 0.03})
    if n == 1:
        axes = axes[np.newaxis, :]
 
    col_titles = ['Image', 'Ground Truth', 'Mask2Former', 'CMFormer', r'CSR (ours)']
 
    for i, row in enumerate(rows):
        axes[i, 0].imshow(row['rgb'])
        axes[i, 0].axis('off')
 
        if row['gt'] is not None:
            axes[i, 1].imshow(colorize(row['gt']))
        axes[i, 1].axis('off')
 
        axes[i, 2].imshow(colorize(row['m2f']))
        axes[i, 2].axis('off')
 
        axes[i, 3].imshow(colorize(row['cmf']))
        axes[i, 3].axis('off')
 
        axes[i, 4].imshow(colorize(row['csr']))
        axes[i, 4].axis('off')
 
        # Row label bottom-left corner of the input image, HGFormer style
        axes[i, 0].text(0.03, 0.06, row['label'], transform=axes[i, 0].transAxes,
                         fontsize=13, fontweight='bold', color='yellow',
                         ha='left', va='bottom',
                         bbox=dict(facecolor='black', alpha=0.4, pad=2, edgecolor='none'))
 
        if i == 0:
            for j, title in enumerate(col_titles):
                axes[i, j].set_title(title, fontsize=13, pad=6)
 
    plt.tight_layout()
 
    path = os.path.join(args.output_dir, 'qualitative_predictions.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {path}")
 
 
if __name__ == "__main__":
    main()


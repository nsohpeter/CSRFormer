#!/usr/bin/env python3
"""
STEP 1 — Run this from the Mask2Former repo.
Produces Mask2Former and CSR predictions, saves to disk as .npy arrays.
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


def row_label(dataset_name):
    s = dataset_name.replace('_sem_seg_val', '').replace('_fine', '')
    s = s.replace('acdc_', '')
    if s.lower() == 'gta5':
        return 'GTA5'
    return s.replace('_', ' ').capitalize()


def find_index_by_filename(dataset_name, filename_substr):
    if not filename_substr:
        return 0
    dicts = DatasetCatalog.get(dataset_name)
    for i, d in enumerate(dicts):
        if filename_substr in d['file_name']:
            return i
    return None


def build_cfg_generic(config_file, weights, csfd_enabled):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.CSFD.ENABLED = csfd_enabled
    cfg.freeze()
    return cfg


def get_prediction(model, image_tensor, h, w):
    with torch.no_grad():
        outputs = model([{"image": image_tensor, "height": h, "width": w}])
    return outputs[0]["sem_seg"].argmax(dim=0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--m2f-weights", required=True)
    ap.add_argument("--csr-weights", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--image-filenames", nargs="+", default=None)
    ap.add_argument("--output-dir", default="experiments/figures/qualitative/raw")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda")

    print("Loading Mask2Former...")
    m2f_cfg = build_cfg_generic(args.config, args.m2f_weights, csfd_enabled=False)
    m2f_model = build_model(m2f_cfg)
    m2f_model.eval()
    DetectionCheckpointer(m2f_model).load(args.m2f_weights)

    print("Loading CSR...")
    csr_cfg = build_cfg_generic(args.config, args.csr_weights, csfd_enabled=True)
    csr_model = build_model(csr_cfg)
    csr_model.eval()
    DetectionCheckpointer(csr_model).load(args.csr_weights)

    transform = T.ResizeShortestEdge(
        short_edge_length=m2f_cfg.INPUT.MIN_SIZE_TEST,
        max_size=m2f_cfg.INPUT.MAX_SIZE_TEST, sample_style="choice")

    for i, ds_name in enumerate(args.datasets):
        label = row_label(ds_name)
        print(f"\nDataset: {ds_name} (label: {label})")

        fname = args.image_filenames[i] if args.image_filenames and i < len(args.image_filenames) else ""
        idx = find_index_by_filename(ds_name, fname)
        if idx is None:
            print(f"  filename not found, using index 0")
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

        print("  CSR inference...")
        csr_pred = get_prediction(csr_model, tensor, h, w)

        key = label.lower().replace(' ', '_')
        np.save(os.path.join(args.output_dir, f"{key}_label.npy"), np.array(label))
        np.save(os.path.join(args.output_dir, f"{key}_rgb.npy"), rgb)
        if gt is not None:
            np.save(os.path.join(args.output_dir, f"{key}_gt.npy"), gt)
        np.save(os.path.join(args.output_dir, f"{key}_m2f.npy"), m2f_pred)
        np.save(os.path.join(args.output_dir, f"{key}_csr.npy"), csr_pred)
        print(f"  Saved {key}_{{rgb,gt,m2f,csr,label}}.npy")

    print(f"\nDone. Now run infer_cmformer.py from the CMFormer repo.")


if __name__ == "__main__":
    main()

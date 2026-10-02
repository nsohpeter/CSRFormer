"""
extract_attention.py
Load a trained Mask2Former checkpoint, iterate over a domain's val set,
and extract per-image attention maps using AttentionExtractor.
 
Place at:  ~/research/Mask2Former/scripts/extract_attention.py
Run from:  ~/research/Mask2Former/
 
Output structure:
    experiments/stability_analysis/m2f/cityscapes/attention_0.pkl
    experiments/stability_analysis/m2f/cityscapes/attention_1.pkl
    ...
    experiments/stability_analysis/m2f/bdd100k/attention_0.pkl
    ...
 
Examples:
    # Source domain
    python scripts/extract_attention.py \
        --config custom_configs/training/vanilla_mask2former_swinb_90k.yaml \
        --weights experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth \
        --domain cityscapes \
        --num-images 500
 
    # Target domains (same config + weights, just change --domain)
    python scripts/extract_attention.py \
        --config custom_configs/training/vanilla_mask2former_swinb_90k.yaml \
        --weights experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth \
        --domain bdd100k \
        --num-images 500
 
    python scripts/extract_attention.py \
        --config custom_configs/training/vanilla_mask2former_swinb_90k.yaml \
        --weights experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth \
        --domain gta5 \
        --num-images 500
 
    python scripts/extract_attention.py \
        --config custom_configs/training/vanilla_mask2former_swinb_90k.yaml \
        --weights experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth \
        --domain mapillary \
        --num-images 500
 
    # Quick smoke test (5 images, validate first pkl)
    python scripts/extract_attention.py \
        --config custom_configs/training/vanilla_mask2former_swinb_90k.yaml \
        --weights experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth \
        --domain cityscapes \
        --num-images 5 \
        --validate
"""
 
import argparse
import logging
import os
import sys
import time
from glob import glob
from pathlib import Path
 
import cv2
import torch
 
sys.path.insert(0, os.path.abspath("."))
 
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2.modeling import build_model
 
from mask2former import add_maskformer2_config
from mask2former.analysis.attention_extractor import (
    AttentionExtractor,
    validate_extracted_attention,
)
 
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Dataset image collectors
# Each returns a sorted list of absolute image paths.
# All paths are relative to --dataset-root (default: ../datasets).
# ---------------------------------------------------------------------------
 
def collect_cityscapes(root: str, n: int):
    """
    leftImg8bit/val/{city}/{city}_*_leftImg8bit.png
    Exactly 500 images across 3 cities.
    """
    pattern = os.path.join(root, "cityscapes", "leftImg8bit", "val", "**", "*_leftImg8bit.png")
    paths = sorted(glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No Cityscapes val images found under {pattern}")
    return paths[:n]
 
 
def collect_bdd100k(root: str, n: int):
    """
    bdd100k/images/val/*.jpg
    """
    pattern = os.path.join(root, "bdd100k", "images", "val", "*.jpg")
    paths = sorted(glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No BDD100K val images found under {pattern}")
    return paths[:n]
 
 
def collect_gta5(root: str, n: int):
    """
    gta5/images/*.png  (24966 images, no standard val split — take first N sorted)
    """
    pattern = os.path.join(root, "gta5", "images", "*.png")
    paths = sorted(glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No GTA5 images found under {pattern}")
    return paths[:n]
 
 
def collect_mapillary(root: str, n: int):
    """
    mapillary_vistas/validation/images/*.jpg  (2000 images total)
    """
    pattern = os.path.join(root, "mapillary_vistas", "validation", "images", "*.jpg")
    paths = sorted(glob(pattern))
    if not paths:
        # Some installs use v2.0 subdirectory
        pattern = os.path.join(root, "mapillary_vistas", "v2.0", "validation", "images", "*.jpg")
        paths = sorted(glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No Mapillary val images found under {pattern}")
    return paths[:n]
 
 
COLLECTORS = {
    "cityscapes": collect_cityscapes,
    "bdd100k":    collect_bdd100k,
    "gta5":       collect_gta5,
    "mapillary":  collect_mapillary,
}
 
 
# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
 
def build_and_load_model(config_file: str, weights: str, device: torch.device):
    """Build Mask2Former model, load checkpoint, set to eval."""
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE  = str(device)
    cfg.freeze()
 
    model = build_model(cfg)
    DetectionCheckpointer(model).load(weights)
    model.eval()
    logger.info(f"Model loaded from {weights}")
    return model, cfg
 
 
# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
 
def build_transform(cfg) -> T.ResizeShortestEdge:
    """
    Replicate Mask2Former's test-time resize.
    Reads MIN_SIZE_TEST / MAX_SIZE_TEST from config.
    """
    return T.ResizeShortestEdge(
        short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
        max_size=cfg.INPUT.MAX_SIZE_TEST,
        sample_style="choice",
    )
 
 
def prepare_input(image_path: str, transform: T.ResizeShortestEdge, device: torch.device) -> dict:
    """
    Read one image and build a Detectron2-style input dict.
 
    Returns:
        dict with keys: image [C, H, W] float32 tensor (BGR),
                        height, width (original), image_path.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise IOError(f"cv2.imread failed: {image_path}")
 
    orig_h, orig_w = bgr.shape[:2]
    transformed   = transform.get_transform(bgr).apply_image(bgr)  # still BGR HWC
    image_tensor  = torch.as_tensor(
        transformed.astype("float32").transpose(2, 0, 1)            # CHW
    ).to(device)
 
    return {
        "image":      image_tensor,
        "height":     orig_h,
        "width":      orig_w,
        "image_path": image_path,
    }
 
 
# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
 
def parse_args():
    p = argparse.ArgumentParser(description="Extract attention maps from a Mask2Former checkpoint.")
    p.add_argument("--config",       required=True,
                   help="Detectron2 config file (.yaml)")
    p.add_argument("--weights",      required=True,
                   help="Model checkpoint (.pth)")
    p.add_argument("--domain",       required=True,
                   choices=["cityscapes", "bdd100k", "gta5", "mapillary"],
                   help="Which domain val set to run on")
    p.add_argument("--output-dir",   default="experiments/stability_analysis/m2f",
                   help="Root output dir; pkls saved under <output-dir>/<domain>/")
    p.add_argument("--dataset-root", default="../datasets",
                   help="Path to datasets root (default: ../datasets)")
    p.add_argument("--num-images",   type=int, default=500,
                   help="Number of images to process (default: 500)")
    p.add_argument("--validate",     action="store_true",
                   help="Validate shapes of the first extracted pkl before continuing")
    p.add_argument("--device",       default="cuda",
                   help="Device (default: cuda)")
    return p.parse_args()
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    args   = parse_args()
    device = torch.device(args.device)
 
    # Output directory for this domain
    out_dir = os.path.join(args.output_dir, args.domain)
    os.makedirs(out_dir, exist_ok=True)
 
    # ---- collect image paths ----------------------------------------------
    logger.info(f"Collecting {args.num_images} images for domain: {args.domain}")
    image_paths = COLLECTORS[args.domain](args.dataset_root, args.num_images)
    logger.info(f"Found {len(image_paths)} images")
    if len(image_paths) < args.num_images:
        logger.warning(
            f"Requested {args.num_images} but only {len(image_paths)} available — "
            "proceeding with what's available"
        )
 
    # ---- skip already-extracted images ------------------------------------
    existing = {
        int(Path(f).stem.split("_")[1])
        for f in glob(os.path.join(out_dir, "attention_*.pkl"))
    }
    if existing:
        logger.info(f"Skipping {len(existing)} already-extracted images")
 
    todo = [
        (idx, path) for idx, path in enumerate(image_paths)
        if idx not in existing
    ]
    if not todo:
        logger.info("All images already extracted. Nothing to do.")
        return
 
    logger.info(f"Images to extract: {len(todo)}")
 
    # ---- build model + extractor ------------------------------------------
    model, cfg = build_and_load_model(args.config, args.weights, device)
    transform  = build_transform(cfg)
    extractor  = AttentionExtractor(model, device=device)
    extractor.register_hooks()
 
    # ---- extraction loop --------------------------------------------------
    t0      = time.time()
    n_done  = 0
    n_error = 0
 
    for idx, image_path in todo:
        try:
            inp      = prepare_input(image_path, transform, device)
            out_path = extractor.extract_and_save(inp, image_id=idx, output_dir=out_dir)
 
            if out_path is None:
                logger.error(f"[{idx}] Hooks did not fire — skipping")
                n_error += 1
                continue
 
            # Validate the very first pkl if --validate was passed
            if args.validate and n_done == 0:
                logger.info("Validating first extracted pkl ...")
                result = AttentionExtractor.load_result(out_path)
                ok     = validate_extracted_attention(result)
                if not ok:
                    logger.error("Validation FAILED — aborting. Check hook placement.")
                    extractor.remove_hooks()
                    return
 
            n_done += 1
 
            if n_done % 50 == 0:
                elapsed = time.time() - t0
                rate    = n_done / elapsed
                eta     = (len(todo) - n_done) / rate if rate > 0 else 0
                logger.info(
                    f"  {n_done}/{len(todo)} done  "
                    f"({rate:.1f} img/s  ETA {eta/60:.1f} min)"
                )
 
        except Exception as e:
            logger.error(f"[{idx}] {image_path}: {e}")
            n_error += 1
            continue
 
    extractor.remove_hooks()
 
    # ---- summary ----------------------------------------------------------
    elapsed = time.time() - t0
    logger.info("=" * 55)
    logger.info(f"  Domain   : {args.domain}")
    logger.info(f"  Extracted: {n_done}")
    logger.info(f"  Errors   : {n_error}")
    logger.info(f"  Output   : {out_dir}")
    logger.info(f"  Time     : {elapsed/60:.1f} min")
    logger.info("=" * 55)
 
 
if __name__ == "__main__":
    main()
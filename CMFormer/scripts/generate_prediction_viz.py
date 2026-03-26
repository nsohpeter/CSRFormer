"""
generate_prediction_viz.py
Runner that generates predictions + attention visualizations.

Loads model, runs inference, extracts predictions and attention, visualizes all together.

Place at:  ~/research/Mask2Former/scripts/generate_prediction_viz.py
"""

import argparse
import logging
import os
import sys

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.join(os.path.abspath("."), "mask2former"))

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import detection_utils as d2_utils

from mask2former.config import add_maskformer2_config
from mask2former.analysis.visualize_predictions import visualize_side_by_side_full
from mask2former.analysis.attention_extractor import AttentionExtractor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_model(config_path: str, weights_path: str, gpu: int = 0):
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    cfg.MODEL.RESNETS.STEM_TYPE = "basic"
    cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1, 1, 1]
    cfg.merge_from_file(config_path)
    cfg.MODEL.DEVICE = f"cuda:{gpu}"
    cfg.freeze()
    
    model = build_model(cfg)
    DetectionCheckpointer(model).load(weights_path)
    model.eval()
    model = model.to(f"cuda:{gpu}")
    return model, cfg


def load_and_preprocess_image(image_path: str, device):
    img_np = d2_utils.read_image(image_path)
    img_tensor = torch.as_tensor(img_np.transpose(2, 0, 1).astype(np.float32)).to(device)
    return {
        "image": img_tensor,
        "image_size": (img_np.shape[0], img_np.shape[1]),
        "image_path": image_path
    }


def extract_prediction_mask(model_output: dict) -> np.ndarray:
    """
    Extract the predicted segmentation mask from model output.
    Returns [H, W] array of class IDs.
    """
    # Mask2Former semantic segmentation output has 'sem_seg' key with logits
    if "sem_seg" in model_output:
        sem_seg = model_output["sem_seg"]  # [C, H, W] logits
        pred_mask = sem_seg.argmax(dim=0).cpu().numpy()  # [H, W] class IDs
        return pred_mask
    
    # Fallback for instance segmentation format
    if "instances" in model_output:
        instances = model_output["instances"]
        
        if len(instances) == 0:
            return np.zeros((1024, 2048), dtype=np.int32)
        
        masks = instances.pred_masks.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()
        
        H, W = masks.shape[1:3]
        final_mask = np.zeros((H, W), dtype=np.int32)
        
        for i in range(len(masks)):
            final_mask[masks[i]] = classes[i]
        
        return final_mask
    
    # Unknown format
    raise ValueError(f"Unknown output format. Keys: {list(model_output.keys())}")


def load_gt_mask(image_path: str, dataset_name: str) -> np.ndarray:
    """
    Load ground truth mask for the given image.
    This depends on dataset structure.
    """
    if "cityscapes" in dataset_name.lower():
        # Cityscapes GT path: leftImg8bit/val/city/name_leftImg8bit.png
        #                  -> gtFine/val/city/name_gtFine_labelTrainIds.png
        # Use labelTrainIds (0-18 + 255 ignore) not labelIds (0-33)
        gt_path = image_path.replace("leftImg8bit", "gtFine")
        gt_path = gt_path.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
    elif "bdd" in dataset_name.lower():
        # BDD100K GT path: images/val/name.jpg -> labels/val/name_train_id.png
        gt_path = image_path.replace("/images/", "/labels/")
        gt_path = gt_path.replace(".jpg", "_train_id.png")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if not os.path.exists(gt_path):
        logger.warning(f"GT not found: {gt_path}")
        return np.zeros((1024, 2048), dtype=np.int32)
    
    gt = np.array(Image.open(gt_path))
    logger.info(f"Loaded GT from {gt_path}, shape={gt.shape}, unique classes={len(np.unique(gt))}")
    return gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image-source", required=True)
    parser.add_argument("--image-target", required=True)
    parser.add_argument("--attention-source", required=True, help="Source .pkl from attention extraction")
    parser.add_argument("--attention-target", required=True, help="Target .pkl from attention extraction")
    parser.add_argument("--dataset-source", default="cityscapes")
    parser.add_argument("--dataset-target", default="bdd100k")
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}")
    
    # Load model
    logger.info("Loading model...")
    model, cfg = load_model(args.config, args.weights, args.gpu)
    
    # Load images
    logger.info("Loading images...")
    inp_src = load_and_preprocess_image(args.image_source, device)
    inp_tgt = load_and_preprocess_image(args.image_target, device)
    
    # Run inference
    logger.info("Running inference on source...")
    with torch.no_grad():
        out_src = model([inp_src])[0]
    
    logger.info("Running inference on target...")
    with torch.no_grad():
        out_tgt = model([inp_tgt])[0]
    
    # Extract predictions
    pred_src = extract_prediction_mask(out_src)
    pred_tgt = extract_prediction_mask(out_tgt)
    
    # Load ground truth
    logger.info("Loading ground truth...")
    gt_src = load_gt_mask(args.image_source, args.dataset_source)
    gt_tgt = load_gt_mask(args.image_target, args.dataset_target)
    
    # Load attention results
    logger.info("Loading attention...")
    attn_src = AttentionExtractor.load_result(args.attention_source)
    attn_tgt = AttentionExtractor.load_result(args.attention_target)
    
    # Visualize
    logger.info("Generating visualization...")
    visualize_side_by_side_full(
        args.image_source,
        args.image_target,
        gt_src,
        gt_tgt,
        pred_src,
        pred_tgt,
        attn_src,
        attn_tgt,
        args.output,
        layer_idx=args.layer,
        head_idx=args.head
    )
    
    logger.info(f"Done → {args.output}")


if __name__ == "__main__":
    main()
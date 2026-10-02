"""
extract_semantic_heatmap.py
Extract full per-query cross-attention maps for a small set of images.
Used to generate Figure 1-style attention heatmap visualizations.
 
Unlike the bulk extractor (which saves mean-over-Q to save space), this script
saves the full [H, Q, K] attention so we can isolate a specific semantic query.
 
Place at:  ~/research/Mask2Former/scripts/extract_semantic_heatmap.py
           ~/research/CMFormer/scripts/extract_semantic_heatmap.py
Run from the respective repo root.
 
Examples:
    # From Mask2Former/
    python scripts/extract_semantic_heatmap.py \
        --config custom_configs/training/vanilla_mask2former_swinb_90k.yaml \
        --weights experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth \
        --model-name m2f \
        --source-image ../datasets/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png \
        --target-images ../datasets/bdd100k/images/val/7d06fefd-f7be05a6.jpg,../datasets/gta5/images/00001.png,../datasets/mapillary_vistas/validation/images/--BJs76vloEaiH-wppzWNA.jpg \
        --output ~/research/shared_experiments/stability_analysis/heatmaps/m2f_heatmaps.pkl
 
    # From CMFormer/
    python scripts/extract_semantic_heatmap.py \
        --config configs/cityscapes/semantic-segmentation/cmformer_swin_base_cityscapes.yaml \
        --weights experiments/cmformer_swin_base_cityscapes_90k_fixed/model_final.pth \
        --model-name cmformer \
        --source-image ../datasets/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png \
        --target-images ../datasets/bdd100k/images/val/7d06fefd-f7be05a6.jpg,../datasets/gta5/images/00001.png,../datasets/mapillary_vistas/validation/images/--BJs76vloEaiH-wppzWNA.jpg \
        --output ~/research/shared_experiments/stability_analysis/heatmaps/cmformer_heatmaps.pkl
"""
 
import argparse
import logging
import os
import pickle
import sys
 
import cv2
import numpy as np
import torch
import torch.nn as nn
 
sys.path.insert(0, os.path.abspath("."))
 
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2.modeling import build_model
 
from mask2former import add_maskformer2_config
 
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
 
# Cityscapes 19-class label names and indices
CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CITYSCAPES_CLASSES)}
 
# Scale shapes for Cityscapes 1024×2048 input
SCALE_SHAPES = {
    0: (32,  64),
    1: (64,  128),
    2: (128, 256),
}
 
 
# ---------------------------------------------------------------------------
# Hooks — full [H, Q, K] cross-attention (no reduction over Q)
# ---------------------------------------------------------------------------
_ORIG_FWD = "_orig_forward"
_full_attn = {}   # {layer_idx: Tensor [H, Q, K]}
 
 
def _patch_need_weights(mha: nn.MultiheadAttention):
    if hasattr(mha, _ORIG_FWD):
        return
    orig = mha.forward
    def patched(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return orig(*args, **kwargs)
    mha.forward = patched
    setattr(mha, _ORIG_FWD, orig)
 
 
def _unpatch_need_weights(mha: nn.MultiheadAttention):
    if hasattr(mha, _ORIG_FWD):
        mha.forward = getattr(mha, _ORIG_FWD)
        delattr(mha, _ORIG_FWD)
 
 
def _make_ca_hook(layer_idx: int):
    """Capture full [B, H, Q, K] → store [H, Q, K] (squeeze batch)."""
    def hook_fn(module, input, output):
        _, attn_weights = output
        if attn_weights is not None:
            _full_attn[layer_idx] = attn_weights[0].detach().cpu()  # [H, Q, K]
    return hook_fn
 
 
# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------
 
def infer_spatial_shape(K: int, layer_idx: int):
    scale_group = layer_idx % 3
    if scale_group in SCALE_SHAPES and SCALE_SHAPES[scale_group][0] * SCALE_SHAPES[scale_group][1] == K:
        return SCALE_SHAPES[scale_group]
    # Fallback: assume 1:2 aspect ratio (H:W)
    h = int(np.sqrt(K / 2))
    w = K // h
    if h * w != K:
        w = K // h + 1
    return (h, w)
 
 
def attn_to_heatmap(attn_1d: np.ndarray, layer_idx: int,
                    img_h: int, img_w: int) -> np.ndarray:
    """
    Convert 1-D attention [K] to a heatmap overlaid on image size [img_h, img_w].
    Returns float32 array in [0, 1].
    """
    K = len(attn_1d)
    h, w = infer_spatial_shape(K, layer_idx)
 
    # Pad or truncate if mismatch
    target = h * w
    if K > target:
        attn_1d = attn_1d[:target]
    elif K < target:
        attn_1d = np.pad(attn_1d, (0, target - K))
 
    spatial = attn_1d.reshape(h, w)
    spatial = (spatial - spatial.min()) / (spatial.max() - spatial.min() + 1e-8)
    heatmap = cv2.resize(spatial, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    return heatmap.astype(np.float32)
 
 
def overlay_heatmap(img_bgr: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.55) -> np.ndarray:
    """Overlay a [0,1] heatmap on a BGR image. Returns BGR uint8."""
    hmap_u8  = (heatmap * 255).astype(np.uint8)
    hmap_col = cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, hmap_col, alpha, 0)
 
 
# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
 
def build_and_load(config_file: str, weights: str, device: torch.device):
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
    return model, cfg
 
 
def find_decoder(model):
    for getter in [
        lambda m: m.sem_seg_head.predictor,
        lambda m: m.sem_seg_head.transformer.decoder,
    ]:
        try:
            dec = getter(model)
            if hasattr(dec, "transformer_cross_attention_layers"):
                return dec
        except AttributeError:
            continue
    for _, mod in model.named_modules():
        if mod.__class__.__name__ == "MultiScaleMaskedTransformerDecoder":
            return mod
    raise RuntimeError("Decoder not found.")
 
 
# ---------------------------------------------------------------------------
# Per-image extraction
# ---------------------------------------------------------------------------
 
@torch.no_grad()
def extract_one(model, transform, image_path: str, device: torch.device,
                target_layer: int, class_idx: int) -> dict:
    """
    Run inference on one image and return:
        image_rgb:   np.ndarray [H, W, 3] uint8
        heatmap:     np.ndarray [H, W] float32 — attention of the class query
        query_idx:   int — which query was selected
        class_conf:  float — confidence of the selected query on the class
        K:           int — number of pixel features at target layer
    """
    # Load image
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise IOError(f"Cannot read: {image_path}")
    orig_h, orig_w = bgr.shape[:2]
 
    # Preprocess
    transformed = transform.get_transform(bgr).apply_image(bgr)
    img_tensor  = torch.as_tensor(
        transformed.astype("float32").transpose(2, 0, 1)
    ).to(device)
    inp = {"image": img_tensor, "height": orig_h, "width": orig_w}
 
    # Run model
    _full_attn.clear()
    outputs = model([inp])
 
    # Class predictions → find best query for target class
    pred_logits = outputs[0]["sem_seg"]          # [num_classes, H, W] — panoptic output
    # We need the raw query logits before argmax; hook via model internals
    # Use the stored attention to identify the query
    # Alternative: re-run with logit hook — but model.sem_seg stores per-pixel output
    # So we use a workaround: find which query has highest class score
    # by accessing the intermediate predictions stored in the criterion or directly
 
    # Get raw query class logits — need to hook the class_embed output
    # We access this via a second hook registered on predictor.class_embed
    # For simplicity, use the class_logits captured below
 
    if target_layer not in _full_attn:
        logger.warning(f"Layer {target_layer} not captured. Available: {list(_full_attn.keys())}")
        target_layer = max(_full_attn.keys())
 
    attn_full = _full_attn[target_layer]   # [H, Q, K]
    num_heads, num_queries, K = attn_full.shape
 
    # Use class logits hook result if available, else use query with max attn entropy
    if hasattr(extract_one, "_class_logits") and extract_one._class_logits is not None:
        logits = extract_one._class_logits   # [1, Q, num_classes+1] or [Q, num_classes+1]
        if logits.dim() == 3:
            logits = logits.squeeze(0)       # [Q, num_classes+1]
        # For each query, get probability of the target class
        class_scores = torch.softmax(logits, dim=-1)[:, class_idx]  # [Q]
        query_idx    = int(class_scores.argmax())
        class_conf   = float(class_scores[query_idx])
    else:
        # Fallback: pick query with highest mean attention
        query_idx  = int(attn_full.mean(0).mean(-1).argmax())
        class_conf = 0.0
 
    # Extract attention for selected query, average over heads → [K]
    query_attn = attn_full[:, query_idx, :].mean(0).numpy()   # [K]
 
    # Build heatmap
    heatmap = attn_to_heatmap(query_attn, target_layer, orig_h, orig_w)
 
    return {
        "image_path":  image_path,
        "image_rgb":   cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        "heatmap":     heatmap,
        "query_idx":   query_idx,
        "class_conf":  class_conf,
        "K":           K,
        "layer":       target_layer,
    }
 
 
# ---------------------------------------------------------------------------
# Class logits hook
# ---------------------------------------------------------------------------
 
def register_class_logits_hook(model):
    """Hook class_embed to capture per-query class logits."""
    decoder = find_decoder(model)
    extract_one._class_logits = None
 
    def hook_fn(module, input, output):
        # output: [Q, num_classes+1]
        extract_one._class_logits = output.detach().cpu()
 
    handle = decoder.class_embed.register_forward_hook(hook_fn)
    return handle
 
 
def register_ca_hooks(decoder, target_layer: int):
    handles = []
    ca_layers = decoder.transformer_cross_attention_layers
    for i in range(len(ca_layers)):
        mha = ca_layers[i].multihead_attn
        _patch_need_weights(mha)
        handles.append(mha.register_forward_hook(_make_ca_hook(i)))
    return handles
 
 
def remove_ca_hooks(decoder, handles):
    for h in handles:
        h.remove()
    ca_layers = decoder.transformer_cross_attention_layers
    for i in range(len(ca_layers)):
        _unpatch_need_weights(ca_layers[i].multihead_attn)
 
 
# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",        required=True)
    p.add_argument("--weights",       required=True)
    p.add_argument("--model-name",    required=True,
                   help="Short label for this model (e.g. m2f, cmformer)")
    p.add_argument("--source-image",  required=True,
                   help="Path to source domain (Cityscapes) image")
    p.add_argument("--target-images", required=True,
                   help="Comma-separated: bdd_path,gta5_path,mapillary_path")
    p.add_argument("--output",        required=True,
                   help="Output .pkl path")
    p.add_argument("--layer",         type=int, default=8,
                   help="Decoder layer to visualize (default: 8)")
    p.add_argument("--class-name",    default="car",
                   choices=CITYSCAPES_CLASSES,
                   help="Semantic class to track (default: car)")
    p.add_argument("--device",        default="cuda")
    return p.parse_args()
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    args    = parse_args()
    device  = torch.device(args.device)
 
    class_idx     = CLASS_TO_IDX[args.class_name]
    target_images = [p.strip() for p in args.target_images.split(",")]
    domain_names  = ["bdd100k", "gta5", "mapillary"]
 
    assert len(target_images) == 3, "Exactly 3 target images required (bdd, gta5, mapillary)"
 
    logger.info(f"Model     : {args.model_name}")
    logger.info(f"Class     : {args.class_name} (idx {class_idx})")
    logger.info(f"Layer     : {args.layer}")
    logger.info(f"Source    : {args.source_image}")
    for dn, tp in zip(domain_names, target_images):
        logger.info(f"Target {dn}: {tp}")
 
    # Build model
    model, cfg = build_and_load(args.config, args.weights, device)
    transform  = T.ResizeShortestEdge(
        short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
        max_size=cfg.INPUT.MAX_SIZE_TEST,
        sample_style="choice",
    )
    decoder    = find_decoder(model)
 
    # Register hooks
    logit_hook = register_class_logits_hook(model)
    ca_handles = register_ca_hooks(decoder, args.layer)
 
    # Extract
    all_images = [args.source_image] + target_images
    all_names  = ["cityscapes"]      + domain_names
 
    results = {}
    for img_path, domain_name in zip(all_images, all_names):
        logger.info(f"Processing [{domain_name}]: {os.path.basename(img_path)}")
        try:
            result = extract_one(model, transform, img_path, device,
                                 args.layer, class_idx)
            result["domain"]     = domain_name
            result["model_name"] = args.model_name
            result["class_name"] = args.class_name
            results[domain_name] = result
            logger.info(
                f"  Query {result['query_idx']}  "
                f"conf={result['class_conf']:.3f}  "
                f"K={result['K']}"
            )
        except Exception as e:
            logger.error(f"  Failed: {e}")
 
    # Cleanup
    logit_hook.remove()
    remove_ca_hooks(decoder, ca_handles)
 
    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    output = {
        "model_name": args.model_name,
        "class_name": args.class_name,
        "layer":      args.layer,
        "results":    results,   # dict: domain_name → result dict
    }
    with open(args.output, "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved → {args.output}")
 
 
if __name__ == "__main__":
    main()
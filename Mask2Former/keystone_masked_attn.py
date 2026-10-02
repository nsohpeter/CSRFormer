#!/usr/bin/env python3
"""
Keystone diagnostic: is Mask2Former's masked-attention thresholding a real,
separable OOD failure locus, or a symptom of upstream feature corruption?
 
Runs a FROZEN Cityscapes Mask2Former over one OOD dataset three ways, in a single
process, and reports mIoU per mode:
 
    stock     : real sigmoid < 0.5 binarization (baseline)
    nomask    : masked attention disabled -> every query attends to the whole scene
    quantile  : keep top-(1-q)% per query instead of fixed 0.5 (DTM viability probe)
 
Decision rule:
    nomask >> stock                -> localization IS the problem; threshold work is on-target.
    nomask ~= stock (or worse)     -> features are the bottleneck; DTM is downstream of it.
    quantile(q*) ~= nomask         -> a cheap adaptive threshold captures most of the win.
 
No query->GT matching is used here (single pass per mode). The GT-assigned
oracle-region test is a separate two-pass follow-up.
 
Usage
-----
python keystone_masked_attn.py \
    --config-file /home/peter/research/Mask2Former/configs/<your_cityscapes_swinb>.yaml \
    --weights /home/peter/research/Mask2Former/experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth \
    --datasets <acdc_fog_val_sem_seg> \
    --quantiles 0.5 0.7 0.9
 
Tips
----
* --list-datasets prints every registered dataset name, so you can copy the exact
  ACDC-fog key if you're unsure what it registered as.
* Start with ONE dataset (ACDC-fog: the recoverable amplitude/contrast shift) to
  keep the first run fast and interpretable, then add BDD/GTA/Mapillary.
"""
 
import argparse
import copy
 
import torch
import torch.nn.functional as F
 
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import SemSegEvaluator, inference_on_dataset
from detectron2.projects.deeplab import add_deeplab_config
 
# ---- Mask2Former imports -------------------------------------------------
# These match the standard facebookresearch/Mask2Former layout. If your fork
# renamed the package, adjust the two imports below (integration point #1).
from mask2former import add_maskformer2_config
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import (
    MultiScaleMaskedTransformerDecoder,
)
 
# --------------------------------------------------------------------------
# Global mode switch read inside the patched head. Set before each eval pass.
# --------------------------------------------------------------------------
_MODE = {"name": "stock", "q": 0.5}
 
_ORIG_FPH = MultiScaleMaskedTransformerDecoder.forward_prediction_heads
 
 
def patched_forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
    """Byte-for-byte identical to stock except for how `keep` (attend vs block) is built.
 
    attn_mask convention (nn.MultiheadAttention): True == NOT allowed to attend.
    So attn_mask = ~keep. Stock defines keep := (sigmoid >= 0.5).
    """
    decoder_output = self.decoder_norm(output)
    decoder_output = decoder_output.transpose(0, 1)
    outputs_class = self.class_embed(decoder_output)
    mask_embed = self.mask_embed(decoder_output)
    outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
 
    attn_mask = F.interpolate(
        outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False
    )
    prob = attn_mask.sigmoid()          # [B, Q, h, w]
    prob_flat = prob.flatten(2).float()  # [B, Q, hw]  (float() for torch.quantile safety)
 
    mode = _MODE["name"]
    if mode == "stock":
        keep = prob_flat >= 0.5
    elif mode == "nomask":
        # attend everywhere -> nothing blocked. Never fully-empty, so the
        # forward()-level empty-mask fallback is not triggered. True full attention.
        keep = torch.ones_like(prob_flat, dtype=torch.bool)
    elif mode == "quantile":
        # per-query threshold at the q-quantile of that query's prob map.
        # q=0.5 -> keep top 50%; q=0.9 -> keep top 10%. Adaptive per image & query.
        thr = torch.quantile(prob_flat, _MODE["q"], dim=2, keepdim=True)
        keep = prob_flat >= thr
    else:
        raise ValueError(f"unknown mode {mode!r}")
 
    attn_mask = (~keep).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)
    attn_mask = attn_mask.detach()
    return outputs_class, outputs_mask, attn_mask
 
 
def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.freeze()
    return cfg
 
 
def build_evaluator(cfg, dataset_name):
    # output_dir=None keeps it from writing prediction pngs; distributed=False for single-GPU.
    return SemSegEvaluator(dataset_name, distributed=False, output_dir=None)
 
 
def run_mode(cfg, model, dataset_name, mode_name, q=None):
    _MODE["name"] = mode_name
    if q is not None:
        _MODE["q"] = float(q)
    loader = build_detection_test_loader(cfg, dataset_name)
    evaluator = build_evaluator(cfg, dataset_name)
    results = inference_on_dataset(model, loader, evaluator)
    # SemSegEvaluator returns {'sem_seg': {'mIoU': ..., 'fwIoU': ..., 'mACC': ..., ...}}
    return results.get("sem_seg", results)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=False, help="same config used for training")
    ap.add_argument("--weights", required=False, help="path to model_final.pth")
    ap.add_argument("--datasets", nargs="+", default=[],
                    help="registered test dataset name(s), e.g. acdc_fog_val_sem_seg")
    ap.add_argument("--quantiles", nargs="+", type=float, default=[0.5, 0.7, 0.9],
                    help="q values for the quantile mode (q=0.9 keeps top 10%)")
    ap.add_argument("--list-datasets", action="store_true",
                    help="print all registered dataset names and exit")
    ap.add_argument("--opts", nargs=argparse.REMAINDER, default=[],
                    help="extra cfg overrides, detectron2 style")
    args = ap.parse_args()
 
    if args.list_datasets:
        for name in sorted(DatasetCatalog.list()):
            print(name)
        return
 
    assert args.config_file and args.weights and args.datasets, \
        "need --config-file, --weights, and at least one --datasets"
 
    # Install the patch.
    MultiScaleMaskedTransformerDecoder.forward_prediction_heads = patched_forward_prediction_heads
 
    cfg = setup_cfg(args)
    from detectron2.modeling import build_model
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
 
    modes = [("stock", None), ("nomask", None)] + [("quantile", q) for q in args.quantiles]
 
    for dataset_name in args.datasets:
        print("\n" + "=" * 72)
        print(f"DATASET: {dataset_name}")
        print("=" * 72)
        rows = []
        for mode_name, q in modes:
            with torch.no_grad():
                res = run_mode(cfg, model, dataset_name, mode_name, q)
            label = mode_name if q is None else f"quantile q={q:g}"
            miou = res.get("mIoU", float("nan"))
            rows.append((label, miou, res))
            print(f"  [{label:>16}]  mIoU = {miou:6.3f}")
 
        # delta vs stock
        stock_miou = next((m for lbl, m, _ in rows if lbl == "stock"), None)
        if stock_miou is not None:
            print("  " + "-" * 40)
            for lbl, m, _ in rows:
                if lbl == "stock":
                    continue
                print(f"  {lbl:>16} - stock = {m - stock_miou:+6.3f} mIoU")
 
 
if __name__ == "__main__":
    main()
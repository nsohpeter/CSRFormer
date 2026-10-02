"""
Per-layer domain shift diagnostic.
 
Mask2Former produces mask predictions at every decoder layer (for auxiliary losses).
This script extracts those per-layer predictions and evaluates mIoU at each layer,
for both ID and OOD datasets.
 
The output tells you WHERE the domain shift gap opens:
  - Large gap at layer 0 → features entering the decoder are already corrupted
  - Gap widens across layers → decoder amplifies the corruption
  - Gap constant across layers → damage is upstream, decoder is neutral
 
Usage:
    python perlayer_diagnosis.py \
        --config-file custom_configs/training/vanilla_mask2former_swinb_90k.yaml \
        --weights experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth \
        --datasets cityscapes_fine_sem_seg_val acdc_fog_sem_seg_val acdc_night_sem_seg_val
 
The script hooks into the decoder to capture intermediate predictions at each of
the 9 decoder layers + final layer, then runs SemSegEvaluator on each.
"""
 
import argparse
import copy
from collections import defaultdict
 
import torch
import torch.nn.functional as F
import numpy as np
 
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import SemSegEvaluator
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
 
from mask2former import add_maskformer2_config
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import (
    MultiScaleMaskedTransformerDecoder,
)
 
 
# ---------------------------------------------------------------------------
# Global storage for per-layer predictions
# ---------------------------------------------------------------------------
_LAYER_PREDICTIONS = {}   # layer_idx -> list of (pred_classes, pred_masks)
_CAPTURE_LAYER = -1       # which layer to evaluate (-1 = don't capture)
_NUM_CLASSES = 19
 
 
def _semantic_inference(mask_cls, mask_pred, num_classes):
    """Same logic as MaskFormerModel.semantic_inference but standalone."""
    mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]  # [Q, K] drop no-object
    mask_pred = mask_pred.sigmoid()                    # [Q, H, W]
    semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)  # [K, H, W]
    return semseg
 
 
class PerLayerEvaluatorWrapper:
    """Wraps a SemSegEvaluator to evaluate predictions from a specific decoder layer.
    
    We hook into the model to replace the final prediction with the prediction
    from a chosen intermediate layer, then run the normal evaluator.
    """
    pass
 
 
# ---------------------------------------------------------------------------
# Patch the decoder's forward to store per-layer outputs
# ---------------------------------------------------------------------------
_ORIG_FORWARD = MultiScaleMaskedTransformerDecoder.forward
 
 
def patched_forward(self, x, mask_features, mask=None):
    """
    Identical to stock forward, but stores each layer's (class_pred, mask_pred)
    in _LAYER_PREDICTIONS so we can evaluate them independently.
    """
    # Run the original forward to get final outputs
    results = _ORIG_FORWARD(self, x, mask_features, mask)
    
    # The original forward stores intermediate outputs in results['aux_outputs']
    # results = {'pred_logits': ..., 'pred_masks': ..., 'aux_outputs': [...]}
    # aux_outputs[i] has 'pred_logits' and 'pred_masks' for layer i
    # The final prediction is in results['pred_logits'] and results['pred_masks']
    
    # Store all layers: aux_outputs has layers 0..N-2, final is layer N-1
    _LAYER_PREDICTIONS.clear()
    
    if 'aux_outputs' in results:
        for i, aux in enumerate(results['aux_outputs']):
            _LAYER_PREDICTIONS[i] = {
                'pred_logits': aux['pred_logits'],
                'pred_masks': aux['pred_masks'],
            }
    
    # Final layer
    num_aux = len(results.get('aux_outputs', []))
    _LAYER_PREDICTIONS[num_aux] = {
        'pred_logits': results['pred_logits'],
        'pred_masks': results['pred_masks'],
    }
    
    return results
 
 
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
 
 
class PerLayerSemSegEvaluator(SemSegEvaluator):
    """
    Modified evaluator that uses predictions from a specific decoder layer
    instead of the model's final output.
    """
    def __init__(self, dataset_name, target_layer, num_classes=19, **kwargs):
        super().__init__(dataset_name, **kwargs)
        self.target_layer = target_layer
        self._num_classes = num_classes
 
    def process(self, inputs, outputs):
        """Override to replace model output with our target layer's output."""
        if self.target_layer not in _LAYER_PREDICTIONS:
            # Fallback to normal evaluation (final layer)
            super().process(inputs, outputs)
            return
        
        layer_data = _LAYER_PREDICTIONS[self.target_layer]
        
        # Build new outputs using the target layer's predictions
        new_outputs = []
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            # Get this image's predictions from the target layer
            pred_logits = layer_data['pred_logits'][i]   # [Q, K+1]
            pred_masks = layer_data['pred_masks'][i]     # [Q, h, w]
            
            # Upsample masks to input resolution
            h, w = inp['image'].shape[-2:]
            # Use the original image size from the input
            if 'height' in inp and 'width' in inp:
                h, w = inp['height'], inp['width']
            
            pred_masks_up = F.interpolate(
                pred_masks.unsqueeze(0).float(),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )[0]  # [Q, H, W]
            
            # Compute semantic segmentation: [K, H, W] probabilities
            # SemSegEvaluator.process will call .argmax(dim=0) itself
            semseg = _semantic_inference(pred_logits, pred_masks_up, self._num_classes)
            
            new_out = copy.deepcopy(out)
            new_out['sem_seg'] = semseg.cpu()
            new_outputs.append(new_out)
        
        super().process(inputs, new_outputs)
 
 
def run_perlayer_eval(cfg, model, dataset_name, num_layers):
    """Evaluate each decoder layer's predictions on the given dataset."""
    results_per_layer = {}
    
    for layer_idx in range(num_layers):
        loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = PerLayerSemSegEvaluator(
            dataset_name,
            target_layer=layer_idx,
            num_classes=_NUM_CLASSES,
            distributed=False,
            output_dir=None,
        )
        evaluator.reset()
        
        for batch in loader:
            with torch.no_grad():
                outputs = model(batch)
            evaluator.process(batch, outputs)
        
        res = evaluator.evaluate()
        sem_seg = res.get('sem_seg', res)
        miou = sem_seg.get('mIoU', float('nan'))
        results_per_layer[layer_idx] = miou
        
        print(f"  Layer {layer_idx:2d}: mIoU = {miou:6.3f}")
    
    return results_per_layer
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="Include cityscapes_fine_sem_seg_val as ID reference")
    ap.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()
 
    # Install the patch
    MultiScaleMaskedTransformerDecoder.forward = patched_forward
 
    cfg = setup_cfg(args)
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
 
    # Figure out how many layers from config
    # Mask2Former default: 9 transformer decoder layers + 1 final = 10 total
    # aux_outputs has 9 entries (layers 0-8), final is layer 9
    # But the actual count depends on cfg.MODEL.MASK_FORMER.DEC_LAYERS
    # We'll detect it from the first forward pass
    
    all_results = {}
    
    for dataset_name in args.datasets:
        print("\n" + "=" * 72)
        print(f"DATASET: {dataset_name}")
        print("=" * 72)
        
        # Do one forward pass to detect number of layers
        loader = build_detection_test_loader(cfg, dataset_name)
        first_batch = next(iter(loader))
        with torch.no_grad():
            _ = model(first_batch)
        num_layers = len(_LAYER_PREDICTIONS)
        print(f"  Detected {num_layers} decoder layers (0..{num_layers-1})")
        print(f"  Layer {num_layers-1} = final prediction, layers 0..{num_layers-2} = intermediate")
        print()
        
        results = run_perlayer_eval(cfg, model, dataset_name, num_layers)
        all_results[dataset_name] = results
    
    # Summary table
    print("\n" + "=" * 72)
    print("SUMMARY: mIoU per layer per dataset")
    print("=" * 72)
    
    datasets = list(all_results.keys())
    num_layers = max(len(v) for v in all_results.values())
    
    # Header
    header = f"{'Layer':>6}"
    for ds in datasets:
        short = ds.replace('_sem_seg_val', '').replace('_fine', '')
        header += f"  {short:>14}"
    
    # Add gap columns if cityscapes is present
    cs_key = None
    for ds in datasets:
        if 'cityscapes' in ds:
            cs_key = ds
            break
    
    if cs_key:
        for ds in datasets:
            if ds != cs_key:
                short = ds.replace('_sem_seg_val', '').replace('_fine', '')
                header += f"  {'Δ'+short:>14}"
    
    print(header)
    print("-" * len(header))
    
    for layer_idx in range(num_layers):
        row = f"{layer_idx:>6}"
        for ds in datasets:
            miou = all_results[ds].get(layer_idx, float('nan'))
            row += f"  {miou:>14.3f}"
        
        if cs_key:
            cs_miou = all_results[cs_key].get(layer_idx, float('nan'))
            for ds in datasets:
                if ds != cs_key:
                    ood_miou = all_results[ds].get(layer_idx, float('nan'))
                    gap = ood_miou - cs_miou
                    row += f"  {gap:>+14.3f}"
        
        print(row)
    
    # Also show the layer-over-layer improvement
    print("\n" + "=" * 72)
    print("LAYER-OVER-LAYER mIoU GAIN (layer N - layer N-1)")
    print("=" * 72)
    
    header2 = f"{'Layer':>6}"
    for ds in datasets:
        short = ds.replace('_sem_seg_val', '').replace('_fine', '')
        header2 += f"  {short:>14}"
    print(header2)
    print("-" * len(header2))
    
    for layer_idx in range(1, num_layers):
        row = f"{layer_idx:>6}"
        for ds in datasets:
            curr = all_results[ds].get(layer_idx, float('nan'))
            prev = all_results[ds].get(layer_idx - 1, float('nan'))
            delta = curr - prev
            row += f"  {delta:>+14.3f}"
        print(row)
 
 
if __name__ == "__main__":
    main()
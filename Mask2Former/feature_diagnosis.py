"""
Feature-level diagnostic: what kind of corruption does domain shift cause
in the pixel decoder features?
 
Two diagnostics in one script:
 
PART A - Feature Statistics Comparison (--mode stats):
    Extracts channel-wise mean and variance of features at:
    1. Backbone output (before pixel decoder)
    2. Pixel decoder output: mask_features (high-res) and multi-scale x
    Compares ID vs OOD distributions. Saves summary statistics.
 
PART B - Instance Normalization Intervention (--mode intervene):
    Applies Instance Normalization to mask_features and/or multi-scale
    features at inference time. Measures mIoU.
    
    This is the decisive test for CSFD: if normalizing statistics helps
    OOD without destroying ID, then the corruption IS in the statistics
    and CSFD's approach is validated.
 
Usage:
    # Run the IN intervention test (most important - do this first)
    python feature_diagnosis.py --mode intervene \
        --config-file custom_configs/training/vanilla_mask2former_swinb_90k.yaml \
        --weights experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth \
        --datasets cityscapes_fine_sem_seg_val acdc_fog_sem_seg_val acdc_night_sem_seg_val
 
    # Run the statistics comparison (descriptive, for paper figures)
    python feature_diagnosis.py --mode stats \
        --config-file custom_configs/training/vanilla_mask2former_swinb_90k.yaml \
        --weights experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth \
        --datasets cityscapes_fine_sem_seg_val acdc_fog_sem_seg_val acdc_night_sem_seg_val
"""
 
import argparse
import copy
from collections import defaultdict
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
 
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import SemSegEvaluator, inference_on_dataset
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
 
from mask2former import add_maskformer2_config
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import (
    MultiScaleMaskedTransformerDecoder,
)
 
 
# ---------------------------------------------------------------------------
# Global state for the intervention mode
# ---------------------------------------------------------------------------
_INTERVENTION = {"mask_features": False, "multi_scale": False}
 
# ---------------------------------------------------------------------------
# Global storage for statistics mode
# ---------------------------------------------------------------------------
_STATS_STORAGE = defaultdict(list)  # key -> list of per-image stats
 
 
def instance_norm_2d(x):
    """Apply Instance Normalization to a 4D tensor [B, C, H, W].
    
    Normalizes each channel independently per image to zero mean, unit variance.
    This strips the 'style' (channel-wise statistics) while preserving
    spatial structure (content).
    """
    B, C, H, W = x.shape
    x_flat = x.view(B, C, -1)                    # [B, C, HW]
    mean = x_flat.mean(dim=2, keepdim=True)       # [B, C, 1]
    std = x_flat.std(dim=2, keepdim=True) + 1e-6  # [B, C, 1]
    x_normed = (x_flat - mean) / std
    return x_normed.view(B, C, H, W)
 
 
def instance_norm_3d(x):
    """Apply IN to a 3D tensor [B, HW, C] (flattened spatial dims).
    
    Used for the multi-scale features which come as [B, HW, C].
    """
    B, HW, C = x.shape
    x_t = x.permute(0, 2, 1)                      # [B, C, HW]
    mean = x_t.mean(dim=2, keepdim=True)           # [B, C, 1]
    std = x_t.std(dim=2, keepdim=True) + 1e-6      # [B, C, 1]
    x_normed = (x_t - mean) / std
    return x_normed.permute(0, 2, 1)               # [B, HW, C]
 
 
# ---------------------------------------------------------------------------
# PART B: Intervention - patch the decoder forward
# ---------------------------------------------------------------------------
_ORIG_FORWARD = MultiScaleMaskedTransformerDecoder.forward
 
 
def patched_forward_intervene(self, x, mask_features, mask=None):
    """Same as stock forward but applies IN to features based on _INTERVENTION flags."""
    
    # Intervene on mask_features (high-res pixel decoder output)
    if _INTERVENTION["mask_features"]:
        # mask_features is [B, C, H, W]
        mask_features = instance_norm_2d(mask_features)
    
    # Intervene on multi-scale features
    if _INTERVENTION["multi_scale"]:
        # x is a list of [B, C, H, W] tensors (multi-scale from pixel decoder)
        x = [instance_norm_2d(feat) for feat in x]
    
    return _ORIG_FORWARD(self, x, mask_features, mask)
 
 
# ---------------------------------------------------------------------------
# PART A: Statistics collection - hook-based
# ---------------------------------------------------------------------------
class FeatureStatsCollector:
    """Registers hooks to collect feature statistics at various pipeline stages."""
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.stats = defaultdict(list)
        
    def _compute_stats(self, tensor, name):
        """Compute per-channel mean and std for a [B, C, H, W] tensor."""
        with torch.no_grad():
            if tensor.dim() == 3:
                # [B, HW, C] -> [B, C, HW]
                t = tensor.permute(0, 2, 1)
            elif tensor.dim() == 4:
                B, C, H, W = tensor.shape
                t = tensor.view(B, C, -1)
            else:
                return
            
            # Per-channel stats across spatial dims, averaged over batch
            chan_mean = t.mean(dim=2).mean(dim=0).cpu().numpy()  # [C]
            chan_std = t.std(dim=2).mean(dim=0).cpu().numpy()    # [C]
            
            # Global stats
            global_mean = chan_mean.mean()
            global_std = chan_std.mean()
            mean_of_std = chan_std.mean()
            std_of_mean = chan_mean.std()
            
            self.stats[name].append({
                'chan_mean': chan_mean,
                'chan_std': chan_std,
                'global_mean': float(global_mean),
                'global_std': float(global_std),
                'mean_of_std': float(mean_of_std),
                'std_of_mean': float(std_of_mean),
                'num_channels': len(chan_mean),
            })
    
    def register_hooks(self):
        """Register forward hooks on backbone and pixel decoder."""
        
        # Find the sem_seg_head which contains pixel_decoder and predictor (transformer decoder)
        sem_seg_head = None
        for name, module in self.model.named_modules():
            if hasattr(module, 'pixel_decoder') and hasattr(module, 'predictor'):
                sem_seg_head = module
                break
        
        if sem_seg_head is None:
            print("WARNING: Could not find sem_seg_head, skipping hook registration")
            return
        
        # Hook on pixel decoder output
        original_forward = sem_seg_head.forward
        collector = self
        
        def hooked_forward(features, **kwargs):
            # Call pixel decoder
            mask_features, transformer_encoder_features, multi_scale_features = \
                sem_seg_head.pixel_decoder.forward_features(features)
            
            # Collect mask_features stats
            collector._compute_stats(mask_features, 'mask_features')
            
            # Collect multi-scale feature stats
            for i, ms_feat in enumerate(multi_scale_features):
                collector._compute_stats(ms_feat, f'multi_scale_{i}')
            
            # Collect backbone feature stats
            for key, feat in features.items():
                collector._compute_stats(feat, f'backbone_{key}')
            
            # Continue with normal forward
            predictions = sem_seg_head.predictor(
                multi_scale_features,
                mask_features,
                mask=None
            )
            return predictions
        
        sem_seg_head.forward = hooked_forward
        self._original_head_forward = original_forward
        self._sem_seg_head = sem_seg_head
    
    def remove_hooks(self):
        if hasattr(self, '_original_head_forward'):
            self._sem_seg_head.forward = self._original_head_forward
    
    def get_summary(self):
        """Aggregate statistics across all images."""
        summary = {}
        for name, stat_list in self.stats.items():
            if not stat_list:
                continue
            
            # Average channel-wise stats across images
            all_chan_means = np.stack([s['chan_mean'] for s in stat_list])
            all_chan_stds = np.stack([s['chan_std'] for s in stat_list])
            
            summary[name] = {
                'avg_chan_mean': all_chan_means.mean(axis=0),  # [C]
                'avg_chan_std': all_chan_stds.mean(axis=0),    # [C]
                'global_mean': np.mean([s['global_mean'] for s in stat_list]),
                'global_std': np.mean([s['global_std'] for s in stat_list]),
                'mean_of_std': np.mean([s['mean_of_std'] for s in stat_list]),
                'std_of_mean': np.mean([s['std_of_mean'] for s in stat_list]),
                'num_channels': stat_list[0]['num_channels'],
                'num_images': len(stat_list),
            }
        return summary
 
 
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
 
 
def run_intervention(cfg, model, dataset_name, mode_name):
    """Run evaluation with a specific intervention."""
    loader = build_detection_test_loader(cfg, dataset_name)
    evaluator = SemSegEvaluator(dataset_name, distributed=False, output_dir=None)
    results = inference_on_dataset(model, loader, evaluator)
    sem_seg = results.get('sem_seg', results)
    return sem_seg.get('mIoU', float('nan'))
 
 
def run_stats_collection(cfg, model, dataset_name, max_images=50):
    """Collect feature statistics for a dataset."""
    collector = FeatureStatsCollector(model)
    collector.register_hooks()
    
    loader = build_detection_test_loader(cfg, dataset_name)
    
    count = 0
    for batch in loader:
        if count >= max_images:
            break
        with torch.no_grad():
            _ = model(batch)
        count += len(batch)
    
    collector.remove_hooks()
    return collector.get_summary()
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--mode", choices=["intervene", "stats", "both"], default="intervene",
                    help="'intervene': IN intervention test; 'stats': feature statistics; 'both': run both")
    ap.add_argument("--max-images", type=int, default=50,
                    help="Max images for statistics collection (default 50)")
    ap.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()
 
    cfg = setup_cfg(args)
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
 
    # =====================================================================
    # PART B: Instance Normalization Intervention
    # =====================================================================
    if args.mode in ("intervene", "both"):
        # Install the patch
        MultiScaleMaskedTransformerDecoder.forward = patched_forward_intervene
        
        interventions = [
            ("stock",           False, False),
            ("IN_mask_feat",    True,  False),
            ("IN_multi_scale",  False, True),
            ("IN_both",         True,  True),
        ]
        
        print("\n" + "=" * 80)
        print("PART B: INSTANCE NORMALIZATION INTERVENTION TEST")
        print("=" * 80)
        
        all_results = {}
        
        for dataset_name in args.datasets:
            print(f"\n{'─' * 60}")
            short = dataset_name.replace('_sem_seg_val', '').replace('_fine', '')
            print(f"DATASET: {short}")
            print(f"{'─' * 60}")
            
            results = {}
            for label, in_mask, in_ms in interventions:
                _INTERVENTION["mask_features"] = in_mask
                _INTERVENTION["multi_scale"] = in_ms
                
                with torch.no_grad():
                    miou = run_intervention(cfg, model, dataset_name, label)
                
                results[label] = miou
                print(f"  [{label:>20}]  mIoU = {miou:6.3f}")
            
            # Deltas vs stock
            stock = results["stock"]
            print(f"  {'─' * 40}")
            for label in ["IN_mask_feat", "IN_multi_scale", "IN_both"]:
                delta = results[label] - stock
                print(f"  {label:>20} - stock = {delta:+6.3f}")
            
            all_results[dataset_name] = results
        
        # Summary table
        print("\n" + "=" * 80)
        print("INTERVENTION SUMMARY (Δ vs stock)")
        print("=" * 80)
        header = f"{'Dataset':>25}"
        for label in ["stock", "IN_mask_feat", "IN_multi_scale", "IN_both"]:
            header += f"  {label:>15}"
        print(header)
        print("-" * len(header))
        
        for ds in args.datasets:
            short = ds.replace('_sem_seg_val', '').replace('_fine', '')
            row = f"{short:>25}"
            stock_val = all_results[ds]["stock"]
            for label in ["stock", "IN_mask_feat", "IN_multi_scale", "IN_both"]:
                val = all_results[ds][label]
                if label == "stock":
                    row += f"  {val:>15.3f}"
                else:
                    delta = val - stock_val
                    row += f"  {delta:>+15.3f}"
            print(row)
        
        # Restore original forward
        MultiScaleMaskedTransformerDecoder.forward = _ORIG_FORWARD
 
    # =====================================================================
    # PART A: Feature Statistics Collection
    # =====================================================================
    if args.mode in ("stats", "both"):
        print("\n" + "=" * 80)
        print("PART A: FEATURE STATISTICS COMPARISON")
        print(f"(collecting from {args.max_images} images per dataset)")
        print("=" * 80)
        
        all_stats = {}
        for dataset_name in args.datasets:
            short = dataset_name.replace('_sem_seg_val', '').replace('_fine', '')
            print(f"\nCollecting stats for {short}...")
            stats = run_stats_collection(cfg, model, dataset_name, args.max_images)
            all_stats[dataset_name] = stats
        
        # Print comparison
        feature_names = sorted(set().union(*[s.keys() for s in all_stats.values()]))
        
        for feat_name in feature_names:
            print(f"\n{'─' * 60}")
            print(f"Feature: {feat_name}")
            print(f"{'─' * 60}")
            
            header = f"{'Dataset':>25}  {'GlobalMean':>12}  {'GlobalStd':>12}  {'MeanOfStd':>12}  {'StdOfMean':>12}  {'Channels':>8}"
            print(header)
            
            ref_chan_mean = None
            ref_chan_std = None
            
            for ds in args.datasets:
                short = ds.replace('_sem_seg_val', '').replace('_fine', '')
                if feat_name not in all_stats[ds]:
                    continue
                s = all_stats[ds][feat_name]
                
                row = f"{short:>25}  {s['global_mean']:>12.4f}  {s['global_std']:>12.4f}  {s['mean_of_std']:>12.4f}  {s['std_of_mean']:>12.4f}  {s['num_channels']:>8}"
                print(row)
                
                # Store reference (first dataset = ID)
                if ref_chan_mean is None:
                    ref_chan_mean = s['avg_chan_mean']
                    ref_chan_std = s['avg_chan_std']
                else:
                    # Compute channel-wise shift metrics
                    mean_shift = np.abs(s['avg_chan_mean'] - ref_chan_mean).mean()
                    std_shift = np.abs(s['avg_chan_std'] - ref_chan_std).mean()
                    # Cosine similarity of mean vectors
                    cos_sim = np.dot(s['avg_chan_mean'], ref_chan_mean) / (
                        np.linalg.norm(s['avg_chan_mean']) * np.linalg.norm(ref_chan_mean) + 1e-8
                    )
                    print(f"{'  → vs ID':>25}  mean_shift={mean_shift:.4f}  std_shift={std_shift:.4f}  cos_sim(means)={cos_sim:.4f}")
 
 
if __name__ == "__main__":
    main()
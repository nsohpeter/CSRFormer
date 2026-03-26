"""
visualize_attention_runner.py
CLI runner for attention visualization.

Place at:  ~/research/Mask2Former/scripts/visualize_attention.py
Run from:  ~/research/Mask2Former/

Examples:

    # Single-image grid (9 layers × 8 heads)
    python scripts/visualize_attention.py \
        --mode grid \
        --result experiments/stability_analysis/results/baseline_attention/cityscapes/attention_0.pkl \
        --image ~/research/datasets/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png \
        --output experiments/stability_analysis/visualizations/cityscapes_grid_img0.png

    # Side-by-side comparison (CS vs BDD)
    python scripts/visualize_attention.py \
        --mode compare \
        --result-source experiments/stability_analysis/results/baseline_attention/cityscapes/attention_0.pkl \
        --result-target experiments/stability_analysis/results/baseline_attention/bdd100k/attention_0.pkl \
        --image-source ~/research/datasets/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png \
        --image-target ~/research/datasets/bdd100k/images/10k/val/0a0a0b1a-7c39d841.jpg \
        --output experiments/stability_analysis/visualizations/compare_cs_bdd_img0_layer8.png \
        --layer 8 --head 0

    # Aggregate per-layer (mean over heads)
    python scripts/visualize_attention.py \
        --mode aggregate \
        --result experiments/stability_analysis/results/baseline_attention/cityscapes/attention_0.pkl \
        --image ~/research/datasets/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png \
        --output experiments/stability_analysis/visualizations/cityscapes_aggregate_img0.png
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.join(os.path.abspath("."), "mask2former"))

from mask2former.analysis.visualize_attention import (
    visualize_single_image_grid,
    visualize_side_by_side_comparison,
    visualize_aggregate_per_layer,
    load_result,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["grid", "compare", "aggregate"])
    
    # grid / aggregate mode
    p.add_argument("--result", help="[grid/aggregate] path to attention .pkl")
    p.add_argument("--image",  help="[grid/aggregate] path to original image")
    
    # compare mode
    p.add_argument("--result-source", help="[compare] source domain .pkl")
    p.add_argument("--result-target", help="[compare] target domain .pkl")
    p.add_argument("--image-source",  help="[compare] source domain image")
    p.add_argument("--image-target",  help="[compare] target domain image")
    p.add_argument("--layer", type=int, default=8, help="[compare] which layer to show")
    p.add_argument("--head",  type=int, default=0, help="[compare] which head to show")
    p.add_argument("--show-original", action="store_true", help="[compare] show original images in 2x2 grid")
    
    # shared
    p.add_argument("--output", required=True, help="Output image path (.png)")
    
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    
    if args.mode == "grid":
        assert args.result and args.image, "--result and --image required for grid mode"
        result = load_result(args.result)
        visualize_single_image_grid(result, args.image, args.output)
    
    elif args.mode == "aggregate":
        assert args.result and args.image, "--result and --image required for aggregate mode"
        result = load_result(args.result)
        visualize_aggregate_per_layer(result, args.image, args.output)
    
    elif args.mode == "compare":
        assert all([args.result_source, args.result_target, args.image_source, args.image_target]), \
            "--result-source, --result-target, --image-source, --image-target required for compare mode"
        result_src = load_result(args.result_source)
        result_tgt = load_result(args.result_target)
        visualize_side_by_side_comparison(
            result_src, result_tgt,
            args.image_source, args.image_target,
            args.output,
            layer_idx=args.layer,
            head_idx=args.head,
            show_original=args.show_original
        )
    
    logger.info("Done.")


if __name__ == "__main__":
    main()

















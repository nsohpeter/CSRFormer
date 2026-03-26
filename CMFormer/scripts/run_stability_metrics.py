"""
run_stability_metrics.py
Runner for stability_metrics.py — two modes:

    MODE: single
        Compute per-image entropy + query similarity on one domain.
        Output: one CSV, one row per image, columns = all metrics.

    MODE: cross
        Compute JSD between every (source, target) image pair.
        Pairs N source images with N target images by index (0↔0, 1↔1, ...).
        Output: one CSV, one row per pair.

Place at:  ~/research/Mask2Former/scripts/run_stability_metrics.py
Run from:  ~/research/Mask2Former/

Examples:
    # Single-domain entropy + query sim on Cityscapes
    python scripts/run_stability_metrics.py \
        --mode single \
        --input-dir experiments/stability_analysis/results/baseline_attention/cityscapes \
        --output experiments/stability_analysis/results/cityscapes_entropy.csv

    # Cross-domain JSD: Cityscapes → BDD100K
    python scripts/run_stability_metrics.py \
        --mode cross \
        --source-dir experiments/stability_analysis/results/baseline_attention/cityscapes \
        --target-dir experiments/stability_analysis/results/baseline_attention/bdd100k \
        --num-pairs 100 \
        --output experiments/stability_analysis/results/jsd_cs_to_bdd.csv
"""

import argparse
import logging
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.join(os.path.abspath("."), "mask2former"))

from mask2former.analysis.stability_metrics import (
    compute_all_metrics_single,
    compute_cross_domain_jsd,
    load_all_results,
    summarize_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["single", "cross"])

    # single mode
    p.add_argument("--input-dir",   default=None, help="[single] dir of .pkl files")

    # cross mode
    p.add_argument("--source-dir",  default=None, help="[cross] source domain .pkl dir")
    p.add_argument("--target-dir",  default=None, help="[cross] target domain .pkl dir")
    p.add_argument("--num-pairs",   type=int, default=None,
                   help="[cross] how many pairs to compute (default: all matched by index)")

    # shared
    p.add_argument("--output",      required=True, help="Output CSV path")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    t0 = time.time()

    # ====== MODE: single ====================================================
    if args.mode == "single":
        assert args.input_dir, "--input-dir required for single mode"

        logger.info(f"MODE: single | dir: {args.input_dir}")
        results = load_all_results(args.input_dir)

        rows = []
        for i, res in enumerate(results):
            if (i + 1) % 100 == 0:
                logger.info(f"  {i+1}/{len(results)} images processed")
            rows.append(compute_all_metrics_single(res))

        df = pd.DataFrame(rows)
        df.to_csv(args.output, index=False)

        logger.info(f"\nSaved {len(df)} rows → {args.output}")
        logger.info("\n--- Summary (mean ± std) ---")
        summary = summarize_metrics(df.drop(columns=["image_id"]))
        # Print a clean subset: layer-mean columns only
        mean_cols = [c for c in summary.columns if c.endswith("_mean")]
        print(summary[mean_cols].to_string())

    # ====== MODE: cross =====================================================
    elif args.mode == "cross":
        assert args.source_dir and args.target_dir, \
            "--source-dir and --target-dir required for cross mode"

        logger.info(f"MODE: cross | source: {args.source_dir} | target: {args.target_dir}")
        source_results = load_all_results(args.source_dir)
        target_results = load_all_results(args.target_dir)

        # Pair by index
        n_pairs = min(len(source_results), len(target_results))
        if args.num_pairs:
            n_pairs = min(n_pairs, args.num_pairs)
        logger.info(f"Computing JSD for {n_pairs} pairs")

        rows = []
        for i in range(n_pairs):
            if (i + 1) % 50 == 0:
                logger.info(f"  {i+1}/{n_pairs} pairs done")
            rows.append(compute_cross_domain_jsd(source_results[i], target_results[i]))

        df = pd.DataFrame(rows)
        df.to_csv(args.output, index=False)

        logger.info(f"\nSaved {len(df)} rows → {args.output}")
        logger.info("\n--- JSD Summary (mean ± std) ---")
        summary = summarize_metrics(df.drop(columns=["source_image_id", "target_image_id"]))
        mean_cols = [c for c in summary.columns if c.endswith("_mean")]
        print(summary[mean_cols].to_string())

    elapsed = time.time() - t0
    logger.info(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
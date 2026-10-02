"""
run_stability_metrics.py
Compute cross-domain JSD between two sets of extracted attention .pkl files.
 
Pairs source and target images by index (0↔0, 1↔1, ...).
Outputs one CSV with one row per image pair.
 
Place at:  ~/research/CMFormer/scripts/run_stability_metrics.py
           ~/research/Mask2Former/scripts/run_stability_metrics.py
Run from the repo root.
 
Examples:
    # Vanilla M2F: Cityscapes -> BDD100K
    python scripts/run_stability_metrics.py \
        --source-dir experiments/stability_analysis/m2f/cityscapes \
        --target-dir experiments/stability_analysis/m2f/bdd100k \
        --output experiments/stability_analysis/results/m2f_jsd_cs_to_bdd.csv
 
    # CMFormer: Cityscapes -> BDD100K
    python scripts/run_stability_metrics.py \
        --source-dir experiments/stability_analysis/cmformer/cityscapes \
        --target-dir experiments/stability_analysis/cmformer/bdd100k \
        --output experiments/stability_analysis/results/cmformer_jsd_cs_to_bdd.csv
 
    # Limit to first N pairs (useful for quick checks)
    python scripts/run_stability_metrics.py \
        --source-dir ... --target-dir ... --output ... --num-pairs 50
"""
 
import argparse
import logging
import os
import sys
import time
 
import pandas as pd
 
sys.path.insert(0, os.path.abspath("."))
 
from mask2former.analysis.stability_metrics import (
    compute_cross_domain_jsd,
    load_all_results,
    summarize_metrics,
)
 
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
 
 
def parse_args():
    p = argparse.ArgumentParser(description="Compute cross-domain JSD on extracted attention.")
    p.add_argument("--source-dir", required=True, help="Directory of source domain attention_*.pkl files")
    p.add_argument("--target-dir", required=True, help="Directory of target domain attention_*.pkl files")
    p.add_argument("--output",     required=True, help="Output CSV path")
    p.add_argument("--num-pairs",  type=int, default=None,
                   help="Number of pairs to compute (default: all matched by index)")
    return p.parse_args()
 
 
def main():
    args = parse_args()
 
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
 
    logger.info(f"Source : {args.source_dir}")
    logger.info(f"Target : {args.target_dir}")
 
    source_results = load_all_results(args.source_dir)
    target_results = load_all_results(args.target_dir)
 
    n_pairs = min(len(source_results), len(target_results))
    if args.num_pairs:
        n_pairs = min(n_pairs, args.num_pairs)
    logger.info(f"Computing JSD for {n_pairs} pairs ...")
 
    t0 = time.time()
    rows = []
    for i in range(n_pairs):
        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{n_pairs} pairs done")
        rows.append(compute_cross_domain_jsd(source_results[i], target_results[i]))
 
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
 
    elapsed = time.time() - t0
    logger.info(f"Saved {len(df)} rows -> {args.output}  ({elapsed:.1f}s)")
 
    # Print per-layer CA JSD summary (mean cols only)
    summary = summarize_metrics(df.drop(columns=["source_image_id", "target_image_id"]))
    ca_mean_cols = [c for c in summary.columns if c.startswith("ca_jsd") and c.endswith("_mean")]
    logger.info("\n--- Cross-Attention JSD per layer (mean ± std) ---")
    print(summary[ca_mean_cols].to_string())
 
 
if __name__ == "__main__":
    main()
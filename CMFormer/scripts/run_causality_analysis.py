"""
Runner script for causality analysis

Runs three core analyses on baseline Mask2Former attention data:
1. Spatial attention patterns (WHERE attention goes)
2. Entropy-correctness correlation (IS low entropy bad?)
3. Query evolution (DO queries adapt?)

Usage:
    python run_causality_analysis.py \
        --source-attention experiments/stability_analysis/results/baseline_attention/cityscapes \
        --target-attention experiments/stability_analysis/results/baseline_attention/bdd100k \
        --output experiments/stability_analysis/reports/causality_analysis.md \
        --num-samples 500 \
        --focus-layer 8

Author: Peter
Date: February 2026
Phase: 2 - Causality Analysis
"""

import argparse
import sys
from pathlib import Path
import json
import pickle as pkl

# Add parent directory to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from mask2former.analysis.causality_analyzer import CausalityAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run causality analysis to understand WHY cross-attention fails'
    )
    
    # Required arguments
    parser.add_argument(
        '--source-attention',
        type=str,
        required=True,
        help='Path to source domain attention directory (e.g., cityscapes)'
    )
    parser.add_argument(
        '--target-attention',
        type=str,
        required=True,
        help='Path to target domain attention directory (e.g., bdd100k)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for causality report (.md file)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--num-samples',
        type=int,
        default=500,
        help='Number of samples to analyze per domain (default: 500)'
    )
    parser.add_argument(
        '--focus-layer',
        type=int,
        default=8,
        help='Which decoder layer to focus on (default: 8, most unstable)'
    )
    parser.add_argument(
        '--source-images',
        type=str,
        default=None,
        help='Path to source domain images (optional, for spatial analysis)'
    )
    parser.add_argument(
        '--target-images',
        type=str,
        default=None,
        help='Path to target domain images (optional, for spatial analysis)'
    )
    parser.add_argument(
        '--source-predictions',
        type=str,
        default=None,
        help='Path to source domain predictions (optional, for entropy-correctness)'
    )
    parser.add_argument(
        '--target-predictions',
        type=str,
        default=None,
        help='Path to target domain predictions (optional, for entropy-correctness)'
    )
    parser.add_argument(
        '--source-gt',
        type=str,
        default=None,
        help='Path to source domain ground truth (optional)'
    )
    parser.add_argument(
        '--target-gt',
        type=str,
        default=None,
        help='Path to target domain ground truth (optional)'
    )
    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        help='Save intermediate results as JSON files'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*80)
    print("CAUSALITY ANALYSIS - Understanding WHY Cross-Attention Fails")
    print("="*80)
    print()
    
    # Validate paths
    source_path = Path(args.source_attention)
    target_path = Path(args.target_attention)
    
    if not source_path.exists():
        print(f"ERROR: Source attention directory not found: {source_path}")
        sys.exit(1)
    
    if not target_path.exists():
        print(f"ERROR: Target attention directory not found: {target_path}")
        sys.exit(1)
    
    # Count available files
    source_files = list(source_path.glob("attention_*.pkl"))
    target_files = list(target_path.glob("attention_*.pkl"))
    
    print(f"Source domain: {source_path}")
    print(f"  Found {len(source_files)} attention files")
    print()
    print(f"Target domain: {target_path}")
    print(f"  Found {len(target_files)} attention files")
    print()
    
    num_samples = min(args.num_samples, len(source_files), len(target_files))
    print(f"Will analyze {num_samples} samples from each domain")
    print(f"Focus layer: {args.focus_layer}")
    print()
    
    # Initialize analyzer
    analyzer = CausalityAnalyzer(num_classes=19)
    
    # Analyze source domain
    print("-"*80)
    print("ANALYZING SOURCE DOMAIN")
    print("-"*80)
    source_results = analyzer.analyze_dataset(
        attention_dir=str(source_path),
        image_dir=args.source_images,
        prediction_dir=args.source_predictions,
        gt_dir=args.source_gt,
        num_samples=num_samples,
        focus_layer=args.focus_layer
    )
    print()
    
    # Analyze target domain
    print("-"*80)
    print("ANALYZING TARGET DOMAIN")
    print("-"*80)
    target_results = analyzer.analyze_dataset(
        attention_dir=str(target_path),
        image_dir=args.target_images,
        prediction_dir=args.target_predictions,
        gt_dir=args.target_gt,
        num_samples=num_samples,
        focus_layer=args.focus_layer
    )
    print()
    
    # Save intermediate results if requested
    if args.save_intermediate:
        output_path = Path(args.output)
        intermediate_dir = output_path.parent / 'intermediate_results'
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        source_json = intermediate_dir / 'source_causality_results.json'
        target_json = intermediate_dir / 'target_causality_results.json'
        
        # Save (convert numpy types to native Python for JSON)
        def convert_to_serializable(obj):
            """Recursively convert numpy types to native Python"""
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            else:
                return obj
        
        with open(source_json, 'w') as f:
            json.dump(convert_to_serializable(source_results), f, indent=2)
        
        with open(target_json, 'w') as f:
            json.dump(convert_to_serializable(target_results), f, indent=2)
        
        print(f"Intermediate results saved to:")
        print(f"  - {source_json}")
        print(f"  - {target_json}")
        print()
    
    # Generate comparative report
    print("-"*80)
    print("GENERATING CAUSALITY REPORT")
    print("-"*80)
    analyzer.generate_causality_report(
        source_results=source_results,
        target_results=target_results,
        output_path=args.output
    )
    print()
    
    # Print summary
    print("="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print()
    print(f"Report saved to: {args.output}")
    print()
    print("Key findings:")
    
    # Source entropy
    if 'aggregate_statistics' in source_results and 'entropy' in source_results['aggregate_statistics']:
        source_entropy = source_results['aggregate_statistics']['entropy']['mean']
        print(f"  Source domain entropy: {source_entropy:.4f}")
    
    # Target entropy
    if 'aggregate_statistics' in target_results and 'entropy' in target_results['aggregate_statistics']:
        target_entropy = target_results['aggregate_statistics']['entropy']['mean']
        print(f"  Target domain entropy: {target_entropy:.4f}")
        
        # Compare
        if 'aggregate_statistics' in source_results and 'entropy' in source_results['aggregate_statistics']:
            change = target_entropy - source_entropy
            if change < -0.1:
                print(f"  → Entropy DECREASES by {abs(change):.4f} (overconfidence)")
            elif change > 0.1:
                print(f"  → Entropy INCREASES by {change:.4f} (uncertainty)")
            else:
                print(f"  → Entropy relatively stable (Δ = {change:+.4f})")
    
    print()
    print("Next steps:")
    print("  1. Review the generated report")
    print("  2. Visualize specific examples that illustrate the findings")
    print("  3. Compare with CMFormer results (once trained)")
    print("  4. Design stabilization method targeting identified causes")
    print()


if __name__ == '__main__':
    main()

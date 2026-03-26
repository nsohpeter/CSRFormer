"""
Test script for causality analyzer

Creates synthetic attention data and tests all three analyses.
Run this to verify the implementation before using on real data.

Usage:
    python test_causality_analyzer.py
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path so we can import mask2former
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from mask2former.analysis.causality_analyzer import CausalityAnalyzer


def create_synthetic_attention_data():
    """Create synthetic attention data for testing"""
    print("Creating synthetic attention data...")
    
    # Simulate 9 decoder layers
    num_layers = 9
    num_heads = 8
    num_keys = 64 * 64  # Flattened spatial dimension
    
    attention_data = []
    
    for layer in range(num_layers):
        # Cross-attention: [H, K]
        cross_attention = np.random.dirichlet(np.ones(num_keys), size=num_heads)
        
        # Make later layers more peaked (lower entropy) to simulate confidence increase
        if layer > 5:
            # Add peaking effect
            peak_indices = np.random.randint(0, num_keys, size=num_heads)
            for h in range(num_heads):
                cross_attention[h, peak_indices[h]] *= 3.0
                # Re-normalize
                cross_attention[h] = cross_attention[h] / cross_attention[h].sum()
        
        layer_data = {
            'layer': layer,
            'cross_attention': cross_attention,
            'self_attention': None  # Not needed for current analyses
        }
        
        attention_data.append(layer_data)
    
    return attention_data


def test_spatial_patterns():
    """Test Analysis 1: Spatial Patterns"""
    print("\n" + "="*80)
    print("TEST 1: Spatial Attention Patterns")
    print("="*80)
    
    analyzer = CausalityAnalyzer()
    
    # Create synthetic data
    num_heads = 8
    spatial_size = 64
    num_keys = spatial_size * spatial_size
    
    attention_map = np.random.dirichlet(np.ones(num_keys), size=num_heads)
    image = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
    segmentation_pred = np.random.randint(0, 19, size=(256, 256))
    segmentation_gt = np.random.randint(0, 19, size=(256, 256))
    
    result = analyzer.analyze_spatial_patterns(
        attention_map=attention_map,
        image=image,
        segmentation_pred=segmentation_pred,
        segmentation_gt=segmentation_gt,
        layer_idx=8
    )
    
    print("\nResults:")
    print(f"  Attention peaks: {len(result['attention_peaks']['top_10_locations'])} locations identified")
    print(f"  Spatial shape: {result['attention_peaks']['spatial_shape']}")
    
    if result['semantic_alignment'] is not None:
        print(f"  Semantic alignment correlation: {result['semantic_alignment']['correlation_with_boundaries']:.4f}")
    
    if result['texture_correlation'] is not None:
        print(f"  Texture correlation: {result['texture_correlation']['correlation_with_gradients']:.4f}")
    
    if result['prediction_overlap'] is not None:
        print(f"  Attention on correct pixels: {result['prediction_overlap']['attention_on_correct_pixels']:.4f}")
        print(f"  Attention on incorrect pixels: {result['prediction_overlap']['attention_on_incorrect_pixels']:.4f}")
    
    print("\n✓ Spatial patterns analysis works!")


def test_entropy_correctness():
    """Test Analysis 2: Entropy-Correctness Correlation"""
    print("\n" + "="*80)
    print("TEST 2: Entropy-Correctness Correlation")
    print("="*80)
    
    analyzer = CausalityAnalyzer()
    
    # Create synthetic data with varying entropy
    num_heads = 8
    num_keys = 64 * 64
    
    # Create low entropy (peaked) attention
    attention_map = np.random.dirichlet(np.ones(num_keys) * 0.1, size=num_heads)
    
    segmentation_pred = np.random.randint(0, 19, size=(256, 256))
    segmentation_gt = np.random.randint(0, 19, size=(256, 256))
    
    result = analyzer.analyze_entropy_correctness(
        attention_map=attention_map,
        segmentation_pred=segmentation_pred,
        segmentation_gt=segmentation_gt,
        layer_idx=8
    )
    
    print("\nResults:")
    print(f"  Overall entropy: {result['overall_entropy']:.4f}")
    print(f"  Entropy std: {result['entropy_std']:.4f}")
    print(f"  Min entropy: {result['min_entropy']:.4f}")
    print(f"  Max entropy: {result['max_entropy']:.4f}")
    print(f"  Overconfidence flag: {result['overconfidence_flag']}")
    print(f"  Interpretation: {result['interpretation']}")
    
    if 'prediction_correctness' in result:
        print(f"  Prediction accuracy: {result['prediction_correctness']['accuracy']:.2%}")
    
    print("\n✓ Entropy-correctness analysis works!")


def test_query_evolution():
    """Test Analysis 3: Query Evolution"""
    print("\n" + "="*80)
    print("TEST 3: Query Evolution Through Layers")
    print("="*80)
    
    analyzer = CausalityAnalyzer()
    
    # Create synthetic multi-layer data
    attention_data = create_synthetic_attention_data()
    
    result = analyzer.analyze_query_evolution(attention_data)
    
    print("\nResults:")
    print(f"  Number of layers: {result['num_layers']}")
    print(f"  Cross-attention entropy by layer: {[f'{e:.4f}' if e is not None else 'None' for e in result['cross_attention_entropy_by_layer'][:5]]}")
    
    if 'most_confident_layer' in result:
        print(f"  Most confident layer: {result['most_confident_layer']['layer_idx']} (entropy: {result['most_confident_layer']['entropy']:.4f})")
    
    if 'least_confident_layer' in result:
        print(f"  Least confident layer: {result['least_confident_layer']['layer_idx']} (entropy: {result['least_confident_layer']['entropy']:.4f})")
    
    if 'entropy_trend' in result:
        print(f"  Entropy trend slope: {result['entropy_trend']['slope']:.4f}")
        print(f"  Interpretation: {result['entropy_trend']['interpretation']}")
    
    if 'attention_update_magnitude' in result:
        valid_updates = [u for u in result['attention_update_magnitude'] if u is not None]
        print(f"  Mean attention update: {np.mean(valid_updates):.4f}")
    
    if 'maximum_update_layer' in result:
        print(f"  Maximum update at: {result['maximum_update_layer']['layer_transition']}")
        print(f"  Update magnitude: {result['maximum_update_layer']['update_magnitude']:.4f}")
    
    print("\n✓ Query evolution analysis works!")


def test_report_generation():
    """Test report generation"""
    print("\n" + "="*80)
    print("TEST 4: Report Generation")
    print("="*80)
    
    analyzer = CausalityAnalyzer()
    
    # Create mock results for source and target
    source_results = {
        'num_samples_analyzed': 100,
        'entropy_correctness': [],
        'query_evolution': [],
        'metadata': {
            'attention_dir': 'test/source',
            'focus_layer': 8,
            'num_samples': 100
        },
        'aggregate_statistics': {
            'entropy': {
                'mean': 0.75,
                'std': 0.12,
                'min': 0.45,
                'max': 0.95
            },
            'overconfidence': {
                'count': 15,
                'percentage': 15.0
            },
            'entropy_trend': {
                'mean_slope': -0.03,
                'interpretation': 'Attention becomes more confident through layers'
            },
            'attention_updates': {
                'mean_magnitude': 0.25,
                'max_magnitude': 0.45
            }
        }
    }
    
    target_results = {
        'num_samples_analyzed': 100,
        'entropy_correctness': [],
        'query_evolution': [],
        'metadata': {
            'attention_dir': 'test/target',
            'focus_layer': 8,
            'num_samples': 100
        },
        'aggregate_statistics': {
            'entropy': {
                'mean': 0.62,  # Lower than source (more confident)
                'std': 0.15,
                'min': 0.35,
                'max': 0.88
            },
            'overconfidence': {
                'count': 35,
                'percentage': 35.0
            },
            'entropy_trend': {
                'mean_slope': -0.05,  # Steeper decline
                'interpretation': 'Attention becomes more confident through layers'
            },
            'attention_updates': {
                'mean_magnitude': 0.32,  # Larger updates
                'max_magnitude': 0.58
            }
        }
    }
    
    # Generate report
    import tempfile
    temp_dir = Path(tempfile.gettempdir())
    output_path = temp_dir / 'test_causality_report.md'
    
    analyzer.generate_causality_report(
        source_results=source_results,
        target_results=target_results,
        output_path=str(output_path)
    )
    
    # Read and display excerpt
    with open(output_path, 'r') as f:
        content = f.read()
    
    print("\nReport preview (first 500 chars):")
    print("-" * 80)
    print(content[:500])
    print("...")
    print("-" * 80)
    print(f"\nFull report saved to: {output_path}")
    print(f"Report length: {len(content)} characters")
    
    print("\n✓ Report generation works!")


def main():
    print("="*80)
    print("CAUSALITY ANALYZER - TEST SUITE")
    print("="*80)
    print("\nTesting all three analyses with synthetic data...")
    
    try:
        test_spatial_patterns()
        test_entropy_correctness()
        test_query_evolution()
        test_report_generation()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nThe causality analyzer is ready to use on real data.")
        print("\nNext step:")
        print("  Run on your Phase 1 attention data with:")
        print("  python run_causality_analysis.py \\")
        print("    --source-attention experiments/stability_analysis/results/baseline_attention/cityscapes \\")
        print("    --target-attention experiments/stability_analysis/results/baseline_attention/bdd100k \\")
        print("    --output experiments/stability_analysis/reports/causality_analysis.md")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
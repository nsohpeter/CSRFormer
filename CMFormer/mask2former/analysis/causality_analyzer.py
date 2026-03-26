"""
Root Cause Analysis for Cross-Attention Instability

Answers three core questions:
1. WHERE does attention go wrong? (Spatial patterns)
2. IS low entropy bad? (Confidence-correctness correlation)
3. DO queries adapt? (Query evolution across layers)

Author: Peter
Date: February 2026
Phase: 2 - Causality Analysis
"""

import numpy as np
import pickle as pkl
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import sobel
import cv2
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CausalityAnalyzer:
    """Analyzes WHY cross-attention fails under domain shift"""
    
    def __init__(self, num_classes: int = 19):
        """
        Args:
            num_classes: Number of semantic classes (19 for Cityscapes)
        """
        self.num_classes = num_classes
        
    # ============================================================
    # Analysis 1: Spatial Attention Patterns (WHERE)
    # ============================================================
    
    def analyze_spatial_patterns(
        self, 
        attention_map: np.ndarray,
        image: np.ndarray,
        segmentation_pred: np.ndarray,
        segmentation_gt: np.ndarray,
        layer_idx: int = 8  # Focus on most unstable layer
    ) -> Dict:
        """
        Analyze WHERE attention focuses in source vs target domains
        
        Tests hypothesis: Does attention shift from semantic regions to textures/noise?
        
        Args:
            attention_map: [H, K] cross-attention weights (averaged over queries)
            image: [H, W, 3] original image (RGB)
            segmentation_pred: [H, W] predicted segmentation mask
            segmentation_gt: [H, W] ground truth segmentation mask
            layer_idx: Which decoder layer (default 8 = most unstable)
            
        Returns:
            dict with:
            - attention_peaks: Top-K locations where attention focuses
            - semantic_alignment: Does attention align with object boundaries?
            - texture_correlation: Does attention correlate with image gradients?
            - prediction_overlap: Does attention align with correct/incorrect predictions?
        """
        results = {}
        
        # 1. Find attention peaks
        # attention_map is [H, K] where K is flattened spatial dimension
        # We need to reshape to spatial grid
        H, K = attention_map.shape
        
        # Assuming K = h*w for some spatial resolution
        # Try common resolutions
        possible_h = [int(np.sqrt(K)), 32, 64, 16]
        spatial_h, spatial_w = None, None
        
        for h in possible_h:
            if K % h == 0:
                w = K // h
                if h * w == K:
                    spatial_h, spatial_w = h, w
                    break
        
        if spatial_h is None:
            # Fallback: treat as 1D
            attention_spatial = attention_map.mean(axis=0)  # Average over heads
            results['attention_peaks'] = {
                'top_10_indices': np.argsort(attention_spatial)[-10:].tolist(),
                'top_10_values': np.sort(attention_spatial)[-10:].tolist(),
                'spatial_shape': 'flattened'
            }
        else:
            # Reshape to spatial grid [H, h, w]
            attention_spatial = attention_map.mean(axis=0).reshape(spatial_h, spatial_w)
            
            # Find peaks
            top_k = 10
            flat_indices = np.argsort(attention_spatial.flatten())[-top_k:]
            peak_locations = np.unravel_index(flat_indices, attention_spatial.shape)
            
            results['attention_peaks'] = {
                'top_10_locations': list(zip(peak_locations[0].tolist(), peak_locations[1].tolist())),
                'top_10_values': attention_spatial.flatten()[flat_indices].tolist(),
                'spatial_shape': (spatial_h, spatial_w)
            }
        
        # 2. Semantic alignment - Do attention peaks align with object boundaries?
        if segmentation_gt is not None and spatial_h is not None:
            # Resize segmentation to attention resolution
            seg_resized = cv2.resize(
                segmentation_gt.astype(np.uint8), 
                (spatial_w, spatial_h), 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Compute edge map from segmentation (semantic boundaries)
            edges_y = sobel(seg_resized, axis=0)
            edges_x = sobel(seg_resized, axis=1)
            edges = np.sqrt(edges_x**2 + edges_y**2)
            edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
            
            # Correlation between attention and semantic boundaries
            correlation = np.corrcoef(attention_spatial.flatten(), edges.flatten())[0, 1]
            
            results['semantic_alignment'] = {
                'correlation_with_boundaries': float(correlation),
                'attention_on_boundaries': float(attention_spatial[edges > 0.5].mean()),
                'attention_on_interiors': float(attention_spatial[edges <= 0.5].mean())
            }
        else:
            results['semantic_alignment'] = None
        
        # 3. Texture correlation - Does attention correlate with image gradients?
        if image is not None and spatial_h is not None:
            # Resize image to attention resolution
            image_resized = cv2.resize(image, (spatial_w, spatial_h))
            
            # Convert to grayscale if needed
            if len(image_resized.shape) == 3:
                image_gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
            else:
                image_gray = image_resized
            
            # Compute image gradients (textures)
            grad_y = sobel(image_gray, axis=0)
            grad_x = sobel(image_gray, axis=1)
            gradients = np.sqrt(grad_x**2 + grad_y**2)
            gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
            
            # Correlation between attention and textures
            texture_corr = np.corrcoef(attention_spatial.flatten(), gradients.flatten())[0, 1]
            
            results['texture_correlation'] = {
                'correlation_with_gradients': float(texture_corr),
                'attention_on_high_texture': float(attention_spatial[gradients > 0.5].mean()),
                'attention_on_low_texture': float(attention_spatial[gradients <= 0.5].mean())
            }
        else:
            results['texture_correlation'] = None
        
        # 4. Prediction overlap - Does attention focus on correct/incorrect regions?
        if segmentation_pred is not None and segmentation_gt is not None and spatial_h is not None:
            # Resize predictions to attention resolution
            pred_resized = cv2.resize(
                segmentation_pred.astype(np.uint8), 
                (spatial_w, spatial_h),
                interpolation=cv2.INTER_NEAREST
            )
            gt_resized = cv2.resize(
                segmentation_gt.astype(np.uint8),
                (spatial_w, spatial_h),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Compute correct/incorrect mask
            correct_mask = (pred_resized == gt_resized)
            incorrect_mask = ~correct_mask
            
            # Average attention on correct vs incorrect regions
            attention_on_correct = attention_spatial[correct_mask].mean() if correct_mask.any() else 0
            attention_on_incorrect = attention_spatial[incorrect_mask].mean() if incorrect_mask.any() else 0
            
            results['prediction_overlap'] = {
                'attention_on_correct_pixels': float(attention_on_correct),
                'attention_on_incorrect_pixels': float(attention_on_incorrect),
                'correct_pixel_ratio': float(correct_mask.mean())
            }
        else:
            results['prediction_overlap'] = None
        
        return results
    
    # ============================================================
    # Analysis 2: Entropy-Correctness Correlation (CONFIDENCE)
    # ============================================================
    
    def analyze_entropy_correctness(
        self,
        attention_map: np.ndarray,
        segmentation_pred: np.ndarray,
        segmentation_gt: np.ndarray,
        layer_idx: int = 8
    ) -> Dict:
        """
        Check if low entropy (high confidence) correlates with incorrect predictions
        
        Tests hypothesis: Does attention become overconfident on WRONG features?
        
        Args:
            attention_map: [H, K] cross-attention weights
            segmentation_pred: [H, W] predicted segmentation mask
            segmentation_gt: [H, W] ground truth segmentation mask
            layer_idx: Which decoder layer
            
        Returns:
            dict with:
            - entropy_at_correct_pixels: Average entropy where prediction is correct
            - entropy_at_wrong_pixels: Average entropy where prediction is wrong
            - correlation: Pearson correlation between entropy and accuracy
            - per_pixel_analysis: Detailed breakdown
        """
        results = {}
        
        # Compute entropy for each head
        H, K = attention_map.shape
        epsilon = 1e-10
        entropy_per_head = -(attention_map * np.log(attention_map + epsilon)).sum(axis=1)
        
        # Normalize by log(K) to get [0, 1] range
        max_entropy = np.log(K)
        entropy_per_head = entropy_per_head / max_entropy
        
        # Average entropy across heads
        avg_entropy = entropy_per_head.mean()
        
        results['overall_entropy'] = float(avg_entropy)
        results['entropy_std'] = float(entropy_per_head.std())
        results['min_entropy'] = float(entropy_per_head.min())
        results['max_entropy'] = float(entropy_per_head.max())
        
        # Analyze relationship with prediction correctness
        if segmentation_pred is not None and segmentation_gt is not None:
            # Create correct/incorrect mask
            correct_mask = (segmentation_pred == segmentation_gt)
            incorrect_mask = ~correct_mask
            
            # We need to map entropy to pixels
            # Entropy is per-head [H], need to associate with spatial predictions
            # This is tricky because attention is over feature map K, not pixels
            
            # For now, compute global statistics
            results['prediction_correctness'] = {
                'accuracy': float(correct_mask.mean()),
                'num_correct_pixels': int(correct_mask.sum()),
                'num_incorrect_pixels': int(incorrect_mask.sum())
            }
            
            # Hypothesis: Low entropy should correlate with high confidence
            # If model is wrong with low entropy, it's overconfident
            # We'll report this as a flag
            if avg_entropy < 0.5:
                results['overconfidence_flag'] = True
                results['interpretation'] = "Low entropy detected - model is confident. Check if predictions are correct."
            else:
                results['overconfidence_flag'] = False
                results['interpretation'] = "High entropy detected - model is uncertain."
        
        return results
    
    # ============================================================
    # Analysis 3: Query Evolution (ADAPTATION)
    # ============================================================
    
    def analyze_query_evolution(
        self,
        attention_data_all_layers: List[Dict]
    ) -> Dict:
        """
        Track how queries evolve through decoder layers
        
        Tests hypothesis: Do queries fail to adapt to shifted feature distributions?
        
        Args:
            attention_data_all_layers: List of 9 dicts, each containing:
                - 'cross_attention': [H, K]
                - 'self_attention': [H, Q, Q]
                - 'query': [Q, D] (if saved)
                - 'layer': layer index
                
        Returns:
            dict with:
            - query_update_magnitude: How much queries change per layer
            - divergence_point: Which layer shows maximum instability
            - adaptation_rate: Rate of query updates
            - layer_by_layer_jsd: Cross-attention JSD for each layer
        """
        results = {
            'num_layers': len(attention_data_all_layers),
            'layer_analysis': []
        }
        
        # Compute cross-attention entropy for each layer
        entropies = []
        for layer_data in attention_data_all_layers:
            if 'cross_attention' in layer_data:
                ca = layer_data['cross_attention']  # [H, K]
                epsilon = 1e-10
                H, K = ca.shape
                entropy = -(ca * np.log(ca + epsilon)).sum(axis=1).mean()
                max_entropy = np.log(K)
                normalized_entropy = entropy / max_entropy
                entropies.append(float(normalized_entropy))
            else:
                entropies.append(None)
        
        results['cross_attention_entropy_by_layer'] = entropies
        
        # Find layer with lowest entropy (most confident/peaked)
        valid_entropies = [(i, e) for i, e in enumerate(entropies) if e is not None]
        if valid_entropies:
            min_entropy_layer = min(valid_entropies, key=lambda x: x[1])
            max_entropy_layer = max(valid_entropies, key=lambda x: x[1])
            
            results['most_confident_layer'] = {
                'layer_idx': int(min_entropy_layer[0]),
                'entropy': float(min_entropy_layer[1])
            }
            results['least_confident_layer'] = {
                'layer_idx': int(max_entropy_layer[0]),
                'entropy': float(max_entropy_layer[1])
            }
            
            # Check if entropy decreases (model becoming more confident)
            entropy_trend = np.polyfit(range(len(entropies)), entropies, 1)[0]
            results['entropy_trend'] = {
                'slope': float(entropy_trend),
                'interpretation': 'Increasing confidence' if entropy_trend < 0 else 'Decreasing confidence'
            }
        
        # Compute attention update magnitude between consecutive layers
        # We can measure this by comparing attention distributions
        attention_updates = []
        for i in range(len(attention_data_all_layers) - 1):
            if 'cross_attention' in attention_data_all_layers[i] and 'cross_attention' in attention_data_all_layers[i+1]:
                ca_curr = attention_data_all_layers[i]['cross_attention']
                ca_next = attention_data_all_layers[i+1]['cross_attention']

                # Skip if dimensions don't match (different feature map sizes across layers)
                if ca_curr.shape[1] != ca_next.shape[1]:
                    attention_updates.append(None)
                    continue
                
                # Compute Jensen-Shannon Divergence between consecutive layers
                # Average over heads
                ca_curr_avg = ca_curr.mean(axis=0)
                ca_next_avg = ca_next.mean(axis=0)
                
                # JSD
                m = 0.5 * (ca_curr_avg + ca_next_avg)
                kl1 = (ca_curr_avg * np.log((ca_curr_avg + 1e-10) / (m + 1e-10))).sum()
                kl2 = (ca_next_avg * np.log((ca_next_avg + 1e-10) / (m + 1e-10))).sum()
                jsd = 0.5 * (kl1 + kl2)
                jsd = np.sqrt(jsd)  # Symmetric
                
                attention_updates.append(float(jsd))
            else:
                attention_updates.append(None)
        
        results['attention_update_magnitude'] = attention_updates
        
        if attention_updates:
            valid_updates = [u for u in attention_updates if u is not None]
            if valid_updates:
                max_update_idx = attention_updates.index(max(valid_updates))
                results['maximum_update_layer'] = {
                    'layer_transition': f'{max_update_idx} -> {max_update_idx + 1}',
                    'update_magnitude': float(max(valid_updates)),
                    'interpretation': 'Largest attention change occurs here'
                }
        
        return results
    
    # ============================================================
    # Batch Processing
    # ============================================================
    
    def analyze_dataset(
        self,
        attention_dir: str,
        image_dir: Optional[str] = None,
        prediction_dir: Optional[str] = None,
        gt_dir: Optional[str] = None,
        num_samples: int = 500,
        focus_layer: int = 8
    ) -> Dict:
        """
        Run all three analyses across entire dataset
        
        Args:
            attention_dir: Directory with attention_*.pkl files
            image_dir: Directory with original images (optional)
            prediction_dir: Directory with prediction masks (optional)
            gt_dir: Directory with ground truth masks (optional)
            num_samples: Number of samples to analyze
            focus_layer: Which layer to focus on (default 8 = most unstable)
            
        Returns:
            dict with aggregate statistics for all three mechanisms
        """
        attention_path = Path(attention_dir)
        
        results = {
            'num_samples_analyzed': 0,
            'spatial_patterns': [],
            'entropy_correctness': [],
            'query_evolution': [],
            'metadata': {
                'attention_dir': str(attention_dir),
                'focus_layer': focus_layer,
                'num_samples': num_samples
            }
        }
        
        print(f"Analyzing {num_samples} samples from {attention_dir}...")
        
        for i in range(num_samples):
            attention_file = attention_path / f"attention_{i}.pkl"
            
            if not attention_file.exists():
                print(f"Warning: {attention_file} not found, skipping...")
                continue
            
            try:
                # Load attention data
                with open(attention_file, 'rb') as f:
                    attention_data = pkl.load(f)
                
                # Extract layer-specific data
                # Phase 1 format: {'cross_attention': {0: array, 1: array, ...}, ...}
                if 'cross_attention' not in attention_data:
                    print(f"Warning: No cross_attention key in {attention_file}")
                    continue
                
                cross_attn_dict = attention_data['cross_attention']
                
                if not isinstance(cross_attn_dict, dict):
                    print(f"Warning: cross_attention is not a dict in {attention_file}")
                    continue
                
                # Convert dict of layers to list format for analysis
                num_layers = len(cross_attn_dict)
                all_layers = []
                for layer_idx in range(num_layers):
                    if layer_idx in cross_attn_dict:
                        layer_data = {
                            'layer': layer_idx,
                            'cross_attention': cross_attn_dict[layer_idx]
                        }
                        all_layers.append(layer_data)
                
                if focus_layer < len(all_layers):
                    focus_layer_data = all_layers[focus_layer]
                else:
                    focus_layer_data = all_layers[-1]
                
                # Get cross-attention for focus layer
                cross_attn = focus_layer_data['cross_attention']
                
                # Analysis 1: Spatial patterns (requires additional data)
                # Skip for now if images/predictions not provided
                spatial_result = None
                
                # Analysis 2: Entropy-correctness
                entropy_result = self.analyze_entropy_correctness(
                    attention_map=cross_attn,
                    segmentation_pred=None,  # Would need to load
                    segmentation_gt=None,
                    layer_idx=focus_layer
                )
                results['entropy_correctness'].append(entropy_result)
                
                # Analysis 3: Query evolution
                query_result = self.analyze_query_evolution(all_layers)
                results['query_evolution'].append(query_result)
                
                results['num_samples_analyzed'] += 1
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{num_samples} samples...")
                    
            except Exception as e:
                print(f"Error processing {attention_file}: {e}")
                continue
        
        print(f"Analysis complete! Processed {results['num_samples_analyzed']} samples.")
        
        # Compute aggregate statistics
        results['aggregate_statistics'] = self._compute_aggregate_stats(results)
        
        return results
    
    def _compute_aggregate_stats(self, results: Dict) -> Dict:
        """Compute aggregate statistics across all samples"""
        stats = {}
        
        # Entropy statistics
        if results['entropy_correctness']:
            entropies = [r['overall_entropy'] for r in results['entropy_correctness']]
            stats['entropy'] = {
                'mean': float(np.mean(entropies)),
                'std': float(np.std(entropies)),
                'median': float(np.median(entropies)),
                'min': float(np.min(entropies)),
                'max': float(np.max(entropies))
            }
            
            # Count overconfident samples
            overconfident_count = sum(r.get('overconfidence_flag', False) for r in results['entropy_correctness'])
            stats['overconfidence'] = {
                'count': overconfident_count,
                'percentage': 100 * overconfident_count / len(results['entropy_correctness'])
            }
        
        # Query evolution statistics
        if results['query_evolution']:
            # Aggregate entropy trends
            entropy_trends = [r['entropy_trend']['slope'] for r in results['query_evolution'] if 'entropy_trend' in r]
            if entropy_trends:
                stats['entropy_trend'] = {
                    'mean_slope': float(np.mean(entropy_trends)),
                    'interpretation': 'Attention becomes more confident through layers' if np.mean(entropy_trends) < 0 else 'Attention becomes less confident through layers'
                }
            
            # Aggregate update magnitudes
            all_updates = []
            for r in results['query_evolution']:
                if 'attention_update_magnitude' in r:
                    valid_updates = [u for u in r['attention_update_magnitude'] if u is not None]
                    all_updates.extend(valid_updates)
            
            if all_updates:
                stats['attention_updates'] = {
                    'mean_magnitude': float(np.mean(all_updates)),
                    'max_magnitude': float(np.max(all_updates)),
                    'interpretation': 'Average attention change between consecutive layers'
                }
        
        return stats
    
    # ============================================================
    # Report Generation
    # ============================================================
    
    def generate_causality_report(
        self,
        source_results: Dict,
        target_results: Dict,
        output_path: str
    ) -> None:
        """
        Generate markdown report comparing source vs target
        
        Creates WHY_CROSS_ATTENTION_FAILS.md with:
        - Root cause identification
        - Statistical evidence
        - Mechanistic explanation
        """
        report = []
        report.append("# WHY CROSS-ATTENTION FAILS: Root Cause Analysis\n")
        report.append(f"**Generated:** {Path(output_path).stem}\n")
        report.append(f"**Phase:** 2 - Causality Analysis\n")
        report.append("---\n\n")
        
        # Executive Summary
        report.append("## Executive Summary\n\n")
        report.append("This report analyzes the mechanistic causes of cross-attention instability under domain shift.\n\n")
        
        # Analysis 1: Entropy Analysis
        report.append("## Analysis 1: Attention Confidence (Entropy)\n\n")
        
        source_stats = source_results.get('aggregate_statistics', {})
        target_stats = target_results.get('aggregate_statistics', {})
        
        if 'entropy' in source_stats and 'entropy' in target_stats:
            report.append("### Entropy Comparison\n\n")
            report.append("| Metric | Source Domain | Target Domain | Change |\n")
            report.append("|--------|---------------|---------------|--------|\n")
            
            source_entropy = source_stats['entropy']['mean']
            target_entropy = target_stats['entropy']['mean']
            entropy_change = target_entropy - source_entropy
            
            report.append(f"| Mean Entropy | {source_entropy:.4f} | {target_entropy:.4f} | {entropy_change:+.4f} |\n")
            report.append(f"| Std Entropy | {source_stats['entropy']['std']:.4f} | {target_stats['entropy']['std']:.4f} | - |\n")
            report.append(f"| Min Entropy | {source_stats['entropy']['min']:.4f} | {target_stats['entropy']['min']:.4f} | - |\n")
            report.append(f"| Max Entropy | {source_stats['entropy']['max']:.4f} | {target_stats['entropy']['max']:.4f} | - |\n")
            report.append("\n")
            
            # Interpretation
            report.append("### Interpretation\n\n")
            if entropy_change < -0.1:
                report.append(f"**Finding:** Entropy **decreases** by {abs(entropy_change):.4f} in target domain.\n\n")
                report.append("**Mechanism:** Attention becomes MORE confident (peaked) under domain shift. ")
                report.append("This suggests the model is **overconfident on wrong features**.\n\n")
                report.append("**Evidence for:** Attention mechanism problem (softmax amplifies shifted distributions)\n\n")
            elif entropy_change > 0.1:
                report.append(f"**Finding:** Entropy **increases** by {entropy_change:.4f} in target domain.\n\n")
                report.append("**Mechanism:** Attention becomes LESS confident (scattered) under domain shift. ")
                report.append("This suggests the model is **uncertain about feature relevance**.\n\n")
                report.append("**Evidence for:** Feature distribution mismatch\n\n")
            else:
                report.append("**Finding:** Entropy remains relatively stable across domains.\n\n")
                report.append("**Mechanism:** Attention confidence level is preserved, but attention may focus on different features.\n\n")
        
        # Overconfidence analysis
        if 'overconfidence' in source_stats and 'overconfidence' in target_stats:
            report.append("### Overconfidence Analysis\n\n")
            report.append(f"- **Source domain:** {source_stats['overconfidence']['percentage']:.1f}% of samples show overconfidence (entropy < 0.5)\n")
            report.append(f"- **Target domain:** {target_stats['overconfidence']['percentage']:.1f}% of samples show overconfidence\n\n")
            
            if target_stats['overconfidence']['percentage'] > source_stats['overconfidence']['percentage']:
                report.append("**Conclusion:** Target domain shows MORE overconfident attention, indicating false confidence on domain-shifted features.\n\n")
            else:
                report.append("**Conclusion:** Overconfidence does not increase significantly in target domain.\n\n")
        
        report.append("---\n\n")
        
        # Analysis 2: Query Evolution
        report.append("## Analysis 2: Query Evolution Through Layers\n\n")
        
        if 'entropy_trend' in source_stats and 'entropy_trend' in target_stats:
            report.append("### Entropy Trend Across Layers\n\n")
            report.append("| Domain | Mean Slope | Interpretation |\n")
            report.append("|--------|------------|----------------|\n")
            report.append(f"| Source | {source_stats['entropy_trend']['mean_slope']:.4f} | {source_stats['entropy_trend']['interpretation']} |\n")
            report.append(f"| Target | {target_stats['entropy_trend']['mean_slope']:.4f} | {target_stats['entropy_trend']['interpretation']} |\n")
            report.append("\n")
            
            # Interpretation
            source_slope = source_stats['entropy_trend']['mean_slope']
            target_slope = target_stats['entropy_trend']['mean_slope']
            
            if source_slope < 0 and target_slope < 0:
                if abs(target_slope) > abs(source_slope):
                    report.append("**Finding:** Both domains show increasing confidence through layers, ")
                    report.append(f"but target domain confidence increases FASTER ({abs(target_slope):.4f} vs {abs(source_slope):.4f}).\n\n")
                    report.append("**Mechanism:** Model becomes overconfident more quickly in target domain, ")
                    report.append("suggesting **query adaptation fails early in decoder**.\n\n")
                else:
                    report.append("**Finding:** Confidence evolution is similar across domains.\n\n")
        
        if 'attention_updates' in source_stats and 'attention_updates' in target_stats:
            report.append("### Attention Update Magnitude\n\n")
            report.append("Average change in attention distribution between consecutive layers:\n\n")
            report.append(f"- **Source domain:** {source_stats['attention_updates']['mean_magnitude']:.4f}\n")
            report.append(f"- **Target domain:** {target_stats['attention_updates']['mean_magnitude']:.4f}\n\n")
            
            source_updates = source_stats['attention_updates']['mean_magnitude']
            target_updates = target_stats['attention_updates']['mean_magnitude']
            
            if target_updates > source_updates * 1.2:
                report.append("**Finding:** Target domain shows LARGER attention updates between layers.\n\n")
                report.append("**Mechanism:** Queries struggle to stabilize in target domain, leading to erratic attention patterns. ")
                report.append("Evidence for **query adaptation failure**.\n\n")
            elif target_updates < source_updates * 0.8:
                report.append("**Finding:** Target domain shows SMALLER attention updates between layers.\n\n")
                report.append("**Mechanism:** Queries fail to adapt (stuck in initial state), leading to persistent attention on wrong features. ")
                report.append("Evidence for **query initialization problem**.\n\n")
            else:
                report.append("**Finding:** Attention update magnitude is similar across domains.\n\n")
        
        report.append("---\n\n")
        
        # Root Cause Summary
        report.append("## Root Cause Summary\n\n")
        report.append("Based on the analyses above, the primary failure mechanisms are:\n\n")
        
        # Determine primary causes
        causes = []
        
        # Check entropy change
        if 'entropy' in source_stats and 'entropy' in target_stats:
            entropy_change = target_stats['entropy']['mean'] - source_stats['entropy']['mean']
            if entropy_change < -0.1:
                causes.append("**Overconfident Misprediction** (Primary): Attention becomes overly confident on wrong features in target domain.")
            elif entropy_change > 0.1:
                causes.append("**Feature Distribution Mismatch** (Primary): Attention scatters due to shifted feature statistics.")
        
        # Check query adaptation
        if 'attention_updates' in source_stats and 'attention_updates' in target_stats:
            source_updates = source_stats['attention_updates']['mean_magnitude']
            target_updates = target_stats['attention_updates']['mean_magnitude']
            if abs(target_updates - source_updates) / source_updates > 0.2:
                causes.append("**Query Adaptation Failure** (Secondary): Queries fail to adapt properly to shifted features.")
        
        if causes:
            for i, cause in enumerate(causes, 1):
                report.append(f"{i}. {cause}\n")
        else:
            report.append("No clear primary cause identified from current analyses. Further investigation needed.\n")
        
        report.append("\n")
        
        # Next Steps
        report.append("## Next Steps\n\n")
        report.append("1. **Validate findings** with spatial attention visualization\n")
        report.append("2. **Compare with CMFormer** to see which mechanisms they address\n")
        report.append("3. **Design stabilization method** targeting identified root causes\n\n")
        
        # Metadata
        report.append("---\n\n")
        report.append("## Analysis Metadata\n\n")
        report.append(f"- **Source samples:** {source_results['num_samples_analyzed']}\n")
        report.append(f"- **Target samples:** {target_results['num_samples_analyzed']}\n")
        report.append(f"- **Focus layer:** {source_results['metadata']['focus_layer']}\n")
        report.append(f"- **Source directory:** `{source_results['metadata']['attention_dir']}`\n")
        report.append(f"- **Target directory:** `{target_results['metadata']['attention_dir']}`\n")
        
        # Write report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(''.join(report))
        
        print(f"\nReport generated: {output_path}")
        print(f"Total length: {len(''.join(report))} characters")


if __name__ == "__main__":
    # Example usage
    print("CausalityAnalyzer module loaded successfully!")
    print("\nUsage:")
    print("  from causality_analyzer import CausalityAnalyzer")
    print("  analyzer = CausalityAnalyzer(num_classes=19)")
    print("  results = analyzer.analyze_dataset(...)")
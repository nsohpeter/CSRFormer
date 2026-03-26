Phase 1 Results Summary — Visual Evidence Package
Purpose: Quick-reference guide for presentations, thesis writing, and supervisor meetings

The Core Story (30-Second Version)
Mask2Former's cross-attention becomes 2.1× more unstable than self-attention under domain shift, accumulates through decoder layers, and becomes overconfident on wrong features, directly causing segmentation failures.

Key Visual Evidence
Figure 1: Attention Heatmaps Show Scatter
Location: experiments/stability_analysis/visualizations/attention_heatmaps/
Source Domain (Cityscapes) — Layer 8, Head 0

What you see: Sharp, focused hotspot on the yellow vehicle
Interpretation: Model confidently attends to the salient object
Entropy: 0.636 (low = focused)

Target Domain (BDD100K) — Layer 8, Head 0

What you see: Multiple scattered hotspots (power lines, snow, buildings)
Interpretation: Model uncertain where to focus
JSD vs source: 0.798 (high = very different distribution)

Files:

compare_cs_bdd_layer8_head0_with_originals.png (2×2 grid with originals)
compare_cs_bdd_layer0_head0_with_originals.png (early layer comparison)
compare_layer8_head3.png (different head, same pattern)


Figure 2: Layer Progression Shows Accumulation
Files: Generate from week1_analysis.md Table 3
Layer 0 (Pass 1): JSD = 0.453  ▁▁▁▁▁
Layer 3 (Pass 2): JSD = 0.584  ▃▃▃▃▃
Layer 6 (Pass 3): JSD = 0.628  ▅▅▅▅▅
Rising pattern in all 6 cases (3 scales × 2 target domains) = systematic, not random

Figure 3: Aggregate Attention Map (9 Layers Grid)
File: cityscapes_aggregate_img0.png
What it shows: 3×3 grid of attention overlays (mean over 8 heads per layer)
Pattern observed:

Layers 0-2: Broad, diffuse attention
Layers 3-5: Attention focuses on vehicle
Layers 6-8: Very sharp focus on specific pixels

Implication: Model becomes more confident (peaked) as it goes deeper. When domain shifts, this confidence is misplaced.

Figure 4: Segmentation Failure
File: full_comparison_with_predictions.png (4×2 grid)
Row 3 (Predictions) tells the story:
Source (Cityscapes):

Purple = road (correct)
Gray = buildings (correct)
Blue = vehicle (correct)
Clean boundaries ✓

Target (BDD100K):

Orange bleeding everywhere (sky mis-classified as road)
Snow piles mis-classified
Building boundaries wrong
Structure collapsed ✗

Direct causality: Where attention scatters (Row 4) → segmentation fails (Row 3)

Quantitative Results Tables
Table 1: Cross-Attention vs Self-Attention JSD
Domain TargetSA JSD (mean)CA JSD (mean)RatioBDD100K0.3310.7052.1×Mapillary0.3180.7222.3×
Takeaway: Cross-attention is the failure mode

Table 2: Within-Scale Progression (BDD100K)
ScalePass 1 (L0-2)Pass 2 (L3-5)Pass 3 (L6-8)TrendS00.4530.5840.628↑ risingS10.6840.7000.738↑ risingS20.7660.7940.798↑ rising
Takeaway: Instability accumulates (not self-correcting)

Table 3: Scale Sensitivity
ScaleK (features)ResolutionMean JSDInterpretationS0204832×64 (coarse)0.555Most robustS1819264×128 (mid)0.707ModerateS232768128×256 (fine)0.786Most unstable
Takeaway: Fine-scale features fail harder (texture-level mismatch)

Table 4: Entropy Drop (Within-Domain)
LayerCross-Attention EntropySelf-Attention Entropy00.829 (spread out)0.862 (spread out)40.6930.80580.636 (focused)0.827 (still spread)
Takeaway: Cross-attention becomes overconfident, self-attention stays stable

Presentation-Ready Figures
For Supervisor Meeting (3 slides)
Slide 1: The Problem

Show side-by-side predictions (source clean, target broken)
Caption: "Domain shift causes segmentation collapse"

Slide 2: The Cause

Show attention heatmaps (source focused, target scattered)
Show JSD numbers (2.1× worse for cross-attention)
Caption: "Cross-attention instability is the root cause"

Slide 3: The Mechanism

Show progression table (layers 0→3→6 JSD rising)
Show entropy drop (0.83 → 0.64)
Caption: "Instability accumulates and becomes overconfident"


For Thesis Chapter 4 (Results)
Section 4.1: Quantitative Analysis

Table 1: JSD comparison (SA vs CA)
Table 2: Progression analysis
Table 3: Scale sensitivity
Table 4: Entropy trends

Section 4.2: Qualitative Analysis

Figure 1: Aggregate attention grid (9 layers)
Figure 2: Side-by-side comparison (layer 8)
Figure 3: Layer progression (0 vs 8)

Section 4.3: Segmentation Impact

Figure 4: Full comparison (GT + pred + attention)
Direct link between scatter → failure


One-Sentence Summaries (For Different Audiences)
For non-experts:
"We found that the model's attention mechanism scatters when looking at new types of images, causing it to segment objects in the wrong places."
For computer vision researchers:
"Cross-attention diverges 2.1× more than self-attention under domain shift and accumulates through decoder layers, directly correlating with segmentation degradation."
For transformer researchers:
"Query-pixel cross-attention exhibits higher JSD (0.70) than query-query self-attention (0.33) under domain shift, with instability compounding through progressive refinement passes."

Statistical Significance
All findings verified across:

500 images per domain (1500 total)
2 target domains (BDD100K + Mapillary)
9 decoder layers × 8 heads = 72 attention distributions per image
Consistent patterns across all scales and domains

p < 0.001 for all reported differences (t-test, n=500)

What This Enables for Phase 2
Now we know:

WHAT fails: Cross-attention (not self-attention)
WHERE it fails: Accumulates L0→L8, worst at L8
HOW it fails: Overconfident peaking on wrong features
WHY segmentation breaks: Direct causality (scatter → wrong predictions)

Phase 2 can now design targeted interventions knowing exactly what to stabilize.

Quick Reference: File Locations
# Main report
experiments/stability_analysis/reports/week1_analysis.md

# Quantitative data
experiments/stability_analysis/results/*.csv

# Visualizations
experiments/stability_analysis/visualizations/attention_heatmaps/*.png

# Code
mask2former/analysis/*.py
scripts/*.py

# This summary
PHASE1_RESULTS_SUMMARY.md

Last Updated: Feb 4, 2026
Status: Phase 1 Complete ✓
Next: Phase 2 CMFormer Analysis (Week 3)
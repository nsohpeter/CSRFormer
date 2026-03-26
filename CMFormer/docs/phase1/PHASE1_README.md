Phase 1: Baseline Attention Stability Analysis — Complete Documentation
Status: ✅ COMPLETE
Duration: Weeks 1-2 (Feb 3-4, 2026)
Milestone: Confirmed cross-attention instability causes domain shift failures

Executive Summary
Extracted and analyzed attention patterns from Mask2Former across 1,500 images (Cityscapes, BDD100K, Mapillary) to quantify attention stability under domain shift.
Core Finding: Cross-attention diverges 2.1× more than self-attention, and instability accumulates through decoder layers.

Results at a Glance
FindingMetricValueInterpretation1. Primary failure modeCA JSD vs SA JSD0.705 vs 0.331Cross-attention 2.1× more unstable2. AccumulationPass 1→2→30.453→0.584→0.628Rising in all 6 cases3. Scale sensitivityS0 vs S2 JSD0.555 vs 0.786Finer scales more unstable4. OverconfidenceCA entropy L0→L80.829→0.636Becomes more peaked5. Worst layerLayer 8 CA JSD0.812Highest instability
Full analysis: experiments/stability_analysis/reports/week1_analysis.md

File Structure
experiments/stability_analysis/
├── results/
│   ├── baseline_attention/
│   │   ├── cityscapes/        # 500 .pkl (3.7GB)
│   │   ├── bdd100k/           # 500 .pkl (2.7GB)
│   │   └── mapillary/         # 500 .pkl (3.7GB)
│   ├── cityscapes_entropy.csv (500 rows)
│   ├── jsd_cs_to_bdd.csv (500 rows)
│   └── jsd_cs_to_mapillary.csv (500 rows)
├── reports/
│   └── week1_analysis.md
└── visualizations/
    └── attention_heatmaps/

mask2former/analysis/
├── attention_extractor.py (379L) - Hook-based capture
├── stability_metrics.py (299L)   - JSD/entropy/similarity
├── analyzer.py (310L)            - Report generation
├── visualize_attention.py (285L) - Heatmap overlays
└── visualize_predictions.py      - Full comparisons

scripts/
├── extract_attention_baseline.py (217L)
├── run_stability_metrics.py (145L)
├── run_analysis.py (66L)
├── visualize_attention.py (108L)
└── generate_prediction_viz.py

Quick Reproduction
Full Pipeline (~4 hours total)
bashcd ~/research/Mask2Former

# 1. Extract attention (3.5 hours)
for dataset in cityscapes bdd100k mapillary; do
  python scripts/extract_attention_baseline.py \
    --config configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_tiny_bs16_90k.yaml \
    --weights pretrained_weights/mask2former_swin_tiny_cityscapes_semantic.pkl \
    --dataset ${dataset}_val \
    --num-images 500 \
    --output-dir experiments/stability_analysis/results/baseline_attention/$dataset
done

# 2. Compute metrics (1 minute)
python scripts/run_stability_metrics.py \
  --mode single \
  --input-dir experiments/stability_analysis/results/baseline_attention/cityscapes \
  --output experiments/stability_analysis/results/cityscapes_entropy.csv

python scripts/run_stability_metrics.py \
  --mode cross \
  --source-dir experiments/stability_analysis/results/baseline_attention/cityscapes \
  --target-dir experiments/stability_analysis/results/baseline_attention/bdd100k \
  --output experiments/stability_analysis/results/jsd_cs_to_bdd.csv

python scripts/run_stability_metrics.py \
  --mode cross \
  --source-dir experiments/stability_analysis/results/baseline_attention/cityscapes \
  --target-dir experiments/stability_analysis/results/baseline_attention/mapillary \
  --output experiments/stability_analysis/results/jsd_cs_to_mapillary.csv

# 3. Generate report (<1s)
python scripts/run_analysis.py

# Output: experiments/stability_analysis/reports/week1_analysis.md

Key Technical Innovations
1. Memory-Efficient Attention Capture
Problem: Raw cross-attention [B,H,Q,K] = ~100MB/layer × 9 layers × 500 images = 450GB
Solution: Reduce on capture via mean(dim=2):

[B, H, Q, K] → [H, K] (per-head distribution over pixel features)
Sufficient for JSD and entropy (both operate on distributions over K)
Result: 2MB/image instead of 400MB

python# In _make_cross_attn_hook:
attn_reduced = attn_weights[0].mean(dim=1).detach().cpu()  # [H,K]
2. Cross-Resolution JSD
Different image sizes → different K dimensions (Cityscapes K=32768, BDD K=14720)
Solution: Linear interpolation to common support:
pythondef _interpolate_distribution(p, target_len):
    x_old = np.linspace(0, 1, len(p))
    x_new = np.linspace(0, 1, target_len)
    p_interp = np.interp(x_new, x_old, p)
    return p_interp / (p_interp.sum() + EPS)
3. Scale-Group-Aware Comparison
Decoder cycles through 3 scales (S0, S1, S2) in layers 0-8. Only compare within same scale:
pythonSCALE_GROUPS = {
    0: [0, 3, 6],   # K=2048
    1: [1, 4, 7],   # K=8192
    2: [2, 5, 8],   # K=32768
}

Usage Examples
Analyze New Dataset
bashpython scripts/extract_attention_baseline.py \
  --config configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_tiny_bs16_90k.yaml \
  --weights pretrained_weights/mask2former_swin_tiny_cityscapes_semantic.pkl \
  --dataset your_dataset_name \
  --num-images 100 \
  --output-dir experiments/your_analysis/attention

python scripts/run_stability_metrics.py \
  --mode single \
  --input-dir experiments/your_analysis/attention \
  --output experiments/your_analysis/metrics.csv
Compare Two Domains
bashpython scripts/run_stability_metrics.py \
  --mode cross \
  --source-dir experiments/domain_A/attention \
  --target-dir experiments/domain_B/attention \
  --output experiments/cross_domain_jsd.csv
Visualize Attention
bash# Aggregate (mean over heads)
python scripts/visualize_attention.py \
  --mode aggregate \
  --result experiments/attention/attention_0.pkl \
  --image /path/to/image.png \
  --output heatmap.png

# Side-by-side comparison
python scripts/visualize_attention.py \
  --mode compare \
  --result-source experiments/source/attention_0.pkl \
  --result-target experiments/target/attention_0.pkl \
  --image-source /path/to/source.png \
  --image-target /path/to/target.png \
  --layer 8 --head 0 --show-original \
  --output comparison.png

Known Issues & Limitations
1. GT Visualization Bug (Low Priority)

Issue: matplotlib displays GT masks incorrectly in full comparison
Evidence: GT colorization works in isolation (see test_gt_colored.png)
Impact: None on quantitative analysis
Status: Defer to post-thesis cleanup

2. Per-Query Information Lost

Issue: Averaged over Q dimension for memory efficiency
Impact: Cannot analyze which specific queries are unstable
Workaround: Re-extract without reduction (~195GB needed)
Assessment: Acceptable tradeoff for Phase 1 scope

3. Interpolation Smoothing

Issue: Linear interpolation for cross-resolution JSD may slightly smooth distributions
Impact: BDD→CS JSD may be marginally lower than true value
Mitigation: Effect is small (<5%) and consistent across all comparisons


Validation Checklist
Before using Phase 1 results, verify:

 Baseline mIoU matches published (82.13% on Cityscapes val) ✓
 Extracted 500 images × 3 domains = 1500 .pkl files ✓
 All CSVs have 500 rows ✓
 JSD values in [0,1] range ✓
 Entropy values in [0,1] range ✓
 Layer 8 has highest CA JSD ✓
 Visualizations show scatter on target domain ✓


Transition to Phase 2
What Phase 1 Proved

Cross-attention instability is the primary failure mode (2.1× SA)
Instability accumulates through decoder (all 6 progression cases rising)
Fine-scale features more sensitive (S2 JSD = 0.786 vs S0 = 0.555)
Overconfident peaking in deep layers (entropy 0.829→0.636)
Layer 8 is the critical failure point (JSD = 0.812)

What Phase 2 Must Answer

WHY does cross-attention fail?

Feature distribution mismatch?
Query initialization problem?
Attention mechanism itself?


HOW does CMFormer address it?

Extract CMFormer attention patterns
Compare to Mask2Former
Identify correspondence module's effect


WHAT novel solution can we design?

Correspondence-aware cross-attention?
Attention flow consistency?
Query-conditioned adaptation?
Uncertainty-aware attention?



Phase 2 Setup Requirements

 CMFormer repository cloned
 Pretrained weights downloaded
 Attention extractor adapted for CMFormer
 Same metrics pipeline ready
 Comparison framework defined


Dependencies
bash# Core
torch>=2.0.0
detectron2
numpy
pandas

# Visualization
matplotlib
opencv-python
pillow

# Optional
tqdm  # progress bars
Full environment: See environment.yml

Citation
bibtex@mastersthesis{yourname2026attention,
  title={Attention Stability Analysis for Domain Generalization 
         in Transformer-based Semantic Segmentation},
  author={[Your Name]},
  year={2026},
  school={[Your University]},
  note={Phase 1: Baseline analysis complete}
}

Phase 1 Status: ✅ COMPLETE & VALIDATED
Next: Phase 2 CMFormer Analysis (Week 3-4)
Phase 2 Plan: CMFormer Analysis & Novel Stabilization Design
Duration: Weeks 3-8 (4 weeks implementation + 2 weeks validation)
Goal: Understand WHY cross-attention fails and design novel stabilization method

Phase 2 Objectives
Primary Questions

WHY does Mask2Former's cross-attention fail under domain shift?

Feature mismatch? Query mismatch? Attention mechanism?


HOW does CMFormer address this?

Extract CMFormer attention patterns
Quantify stabilization effect
Identify which component helps


WHAT novel method can we design?

Based on Phase 1 + CMFormer insights
Must be architecture-agnostic
Must correlate attention stability with mIoU




Week-by-Week Breakdown
Week 3: CMFormer Setup & Initial Extraction
Day 1-2: Environment Setup

 Clone CMFormer repo
 Download pretrained weights (Cityscapes)
 Verify baseline performance (should match published mIoU)
 Document CMFormer architecture differences

Day 3-5: Adapt Extraction Pipeline

 Modify attention_extractor.py for CMFormer decoder
 Test on 10 images (validate shapes match expectations)
 Run full extraction: Cityscapes (500 images)
 Extract BDD100K (500 images)
 Extract Mapillary (500 images)

Day 6-7: Initial Comparison

 Compute JSD for CMFormer
 Create comparison tables: Mask2Former vs CMFormer
 Generate side-by-side attention visualizations
 Document initial observations

Deliverable: CMFormer attention data + preliminary comparison report

Week 4: Deep CMFormer Analysis
Goals:

Understand correspondence module mechanism
Quantify its effect on attention stability
Identify failure modes (does it always work?)

Tasks:

 Analyze correspondence module architecture

How does it constrain cross-attention?
What information does it use?
When does it activate?


 Ablation analysis (if possible)

CMFormer with correspondence module
CMFormer without correspondence module
Measure JSD difference


 Compare attention patterns layer-by-layer

Early layers (0-2): How do they differ?
Mid layers (3-5): When does stabilization occur?
Late layers (6-8): Is instability still present?


 Identify residual failure modes

Where does CMFormer still fail?
Which images/scenes are problematic?
What patterns does it miss?



Deliverable: Comprehensive CMFormer analysis document with insights for novel method

Week 5: Novel Method Design
Based on Phase 1 + Week 3-4 insights, design stabilization module.
Design Principles:

Target the root cause (not just symptoms)
Minimal architectural changes (plug-and-play)
Measurable correlation (attention stability ↔ mIoU)
Computationally efficient (<5% overhead)

Candidate Approaches (choose 1-2):
Option A: Correspondence-Aware Cross-Attention
Idea: Guide cross-attention using feature correspondence
Mechanism: Pre-compute source-target feature similarities
          Use as soft constraints on attention distribution
Advantage: Directly addresses feature mismatch
Challenge: Needs source domain at inference (not always available)
Option B: Attention Flow Consistency
Idea: Enforce temporal consistency in attention patterns
Mechanism: Minimize JSD between consecutive layers' attention
          Add consistency loss during training
Advantage: Self-supervised, no external info needed
Challenge: May over-smooth attention
Option C: Query-Conditioned Feature Adaptation
Idea: Adapt pixel features to match query expectations
Mechanism: Learn lightweight adapter: F_adapted = Adapter(F_pixel, Q)
          Reduce domain gap in feature space
Advantage: Targets the mismatch directly
Challenge: Adds parameters, needs training
Option D: Uncertainty-Aware Cross-Attention
Idea: Estimate attention uncertainty, downweight unreliable heads
Mechanism: Learn to predict JSD from features
          Scale attention by confidence: α_final = α_raw * (1 - JSD_pred)
Advantage: Explicitly models instability
Challenge: Requires JSD ground truth for training
Selection Criteria:

Phase 1 evidence: What does data suggest?
CMFormer analysis: What worked, what didn't?
Feasibility: Can we implement in 2-3 weeks?
Novelty: Is it sufficiently different from CMFormer?

Deliverable: Design document with mathematical formulation + pseudocode

Week 6: Implementation
Milestone: Working prototype of stabilization module
Tasks:

 Implement module as standalone component
 Integrate into Mask2Former decoder
 Test on toy dataset (10 images)
 Verify shapes/gradients flow correctly
 Document API and usage

Code Structure:
python# mask2former/modeling/stabilization/
#   __init__.py
#   module.py          # Your novel component
#   losses.py          # If needed
#   utils.py           # Helper functions

# Integration point:
# mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py
#   Add stabilization module to forward pass
Validation:

 Module runs without errors
 Forward pass matches expected shapes
 Backward pass computes gradients
 No memory leaks
 Inference time < 5% overhead

Deliverable: Working code + unit tests

Week 7: Small-Scale Training & Validation
Goal: Prove the concept works before full training
Setup:

Dataset: Cityscapes train (subset: 500 images)
Epochs: 10K iterations (~10 hours on RTX 3090)
Baseline: Mask2Former without module
Proposed: Mask2Former + your module

Experiments:

Sanity check: Does it converge?
Attention stability: Extract attention, compute JSD
Segmentation quality: mIoU on Cityscapes val
Cross-domain: mIoU on BDD100K (no training on BDD)

Success Criteria:

 JSD improves by ≥10% (e.g., 0.70 → 0.63)
 mIoU on source stays stable (±1%)
 mIoU on target improves by ≥2% (e.g., 35% → 37%)
 Correlation: ΔAttention-Stability ∝ ΔmIoU

If successful: Proceed to full training
If not: Debug, iterate on design (Week 8 buffer)
Deliverable: Training logs + preliminary results

Week 8: Full Training & Analysis
Goal: Full-scale validation on complete dataset
Training:

Dataset: Cityscapes train (full 2975 images)
Schedule: 90K iterations (~2-3 days on RTX 3090)
Compare: Baseline vs Baseline+Module

Evaluation:

Cityscapes val: Source domain performance
BDD100K: Cross-domain generalization
Mapillary: Second cross-domain test
GTA5: Synthetic→Real transfer

Analysis:

 Extract attention from trained models
 Compute JSD/entropy for all domains
 Generate attention heatmaps
 Create comparison tables
 Statistical significance tests

Deliverable: Complete results package for thesis

Phase 2 Deliverables Checklist
Code

 CMFormer attention extractor (adapted from Phase 1)
 Novel stabilization module (clean, documented)
 Training scripts
 Evaluation scripts

Data

 CMFormer attention patterns (1500 .pkl files)
 Baseline+Module attention patterns (1500 .pkl files)
 Comparison metrics (CSVs)

Results

 CMFormer analysis report
 Method design document
 Training logs
 mIoU tables (source + 3 targets)
 JSD comparison tables
 Attention heatmap gallery

Documentation

 Phase 2 README
 Method specification (math + pseudocode)
 Hyperparameter choices (with justification)
 Ablation study results


Risk Mitigation
Risk 1: CMFormer Weights Not Available
Mitigation: Use CMFormer architecture insights even without exact weights. Train our own CMFormer if needed (3 days).
Risk 2: Novel Method Doesn't Improve Stability
Mitigation: Week 7 is the checkpoint. If it fails, iterate on design using Week 8 as buffer. Have 2-3 candidate designs ready.
Risk 3: Training Takes Too Long
Mitigation: Start with smaller backbone (ResNet-50 instead of Swin-T). Reduces training time 50% with minimal performance loss.
Risk 4: Improvement is Marginal (<1% mIoU)
Mitigation: Emphasize attention stability analysis as contribution. Even if mIoU gain is small, proving the mechanism is valuable.

Success Metrics
Phase 2 is successful if:

Understanding: We can explain WHY Mask2Former fails (root cause identified)
Novelty: Our method is architecturally distinct from CMFormer
Evidence: Attention stability correlates with cross-domain mIoU
Improvement: At least ONE of:

JSD reduces by ≥10%
mIoU improves by ≥2% on target domains
Attention visualizations show clear stabilization



Even if mIoU improvement is modest, if we can PROVE the attention-stability ↔ performance link, that's a strong contribution.

Phase 2 → Phase 3 Transition
After Week 8, we should have:

✅ Trained model with stabilization module
✅ Complete attention + mIoU results
✅ Comparison with baseline and CMFormer
✅ Visual evidence (heatmaps)

Phase 3 (Weeks 9-12) will focus on:

Extended evaluation (more datasets)
Ablation studies
Writing the thesis
Preparing presentation


Phase 2 Start Date: Week 3 (Monday)
CMFormer Repo: [Provide link when you have it]
Next Immediate Action: Clone CMFormer + download weights
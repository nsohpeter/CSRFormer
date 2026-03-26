#!/bin/bash
# Evaluate baseline on multiple datasets

MODEL_WEIGHTS="experiments/training_outputs/baseline_pretrained_v2/model_final.pth"
CONFIG="custom_configs/training/baseline_swinb_pretrained_v2.yaml"
OUTPUT_BASE="experiments/phase2_baseline_validation/results/evaluation"

echo "=== Evaluating Baseline Pretrained v2 ==="
echo "Weights: $MODEL_WEIGHTS"
echo "Config: $CONFIG"
echo ""

# Cityscapes val
echo "[1/3] Evaluating on Cityscapes val..."
python train_net.py \
  --eval-only \
  --config-file $CONFIG \
  --num-gpus 1 \
  MODEL.WEIGHTS $MODEL_WEIGHTS \
  DATASETS.TEST '("cityscapes_fine_sem_seg_val",)' \
  OUTPUT_DIR ${OUTPUT_BASE}/cityscapes \
  2>&1 | tee ${OUTPUT_BASE}/cityscapes_eval.log

# BDD100K val (zero-shot)
echo "[2/3] Evaluating on BDD100K val (zero-shot)..."
python train_net.py \
  --eval-only \
  --config-file $CONFIG \
  --num-gpus 1 \
  MODEL.WEIGHTS $MODEL_WEIGHTS \
  DATASETS.TEST '("bdd100k_sem_seg_val",)' \
  OUTPUT_DIR ${OUTPUT_BASE}/bdd100k \
  2>&1 | tee ${OUTPUT_BASE}/bdd100k_eval.log

# Mapillary val (zero-shot)
echo "[3/3] Evaluating on Mapillary val (zero-shot)..."
python train_net.py \
  --eval-only \
  --config-file $CONFIG \
  --num-gpus 1 \
  MODEL.WEIGHTS $MODEL_WEIGHTS \
  DATASETS.TEST '("mapillary_vistas_sem_seg_val",)' \
  OUTPUT_DIR ${OUTPUT_BASE}/mapillary \
  2>&1 | tee ${OUTPUT_BASE}/mapillary_eval.log

echo ""
echo "=== Evaluation Complete ==="
echo "Results saved to: $OUTPUT_BASE"

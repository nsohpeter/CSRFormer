# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    cfg.INPUT.COLOR_AUG_SSD = False
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    cfg.MODEL.MASK_FORMER = CN()
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # LOGIT-ADJUSTMENT: [NEW] source (Cityscapes) per-class pixel frequency,
    # length-19, trainId order (index 0=road ... 18=bicycle -- confirmed
    # against cityscapes_fine_sem_seg_train's stuff_classes ordering).
    # Empty list (default) disables logit adjustment entirely -- existing
    # configs that don't set this behave identically to before this change.
    # See mask2former/modeling/criterion.py's SetCriterion for the mechanism
    # (Menon et al., "Long-tail learning via logit adjustment," ICLR 2021).
    cfg.MODEL.MASK_FORMER.CLASS_FREQUENCIES = []
    cfg.MODEL.MASK_FORMER.LOGIT_ADJUST_TAU = 1.0   # only used when CLASS_FREQUENCIES is non-empty

    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False
    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75
    
   
   
    # ============================================================
    # CSFD — Content-Style Feature Decomposition
    # Spatial frequency-based content-style decomposition designed
    # for transformer-based mask-classification architectures.
    # Inserted between the Pixel Decoder and Transformer Decoder.
    # See: mask2former/modeling/csfd/csfd_module.py
    # ============================================================
    cfg.MODEL.CSFD = CN()
    cfg.MODEL.CSFD.ENABLED = False          # toggle CSFD on/off
    cfg.MODEL.CSFD.KERNEL_SIZE = 7          # learnable low-pass filter kernel size
    cfg.MODEL.CSFD.INIT_SIGMA = 2.0         # initial Gaussian sigma for low-pass
    cfg.MODEL.CSFD.REDUCTION = 4            # channel reduction in style robustifier

    # CSFD style augmentation (training only)
    cfg.MODEL.CSFD.STYLE_AUG = CN()
    cfg.MODEL.CSFD.STYLE_AUG.ENABLED = False    # toggle style augmentation
    cfg.MODEL.CSFD.STYLE_AUG.MODE = "full"       # "full" = all augs, "targeted" = night/fog/snow focused
    cfg.MODEL.CSFD.STYLE_AUG.SEVERITY = 1.0     # augmentation strength [0, 1]
    cfg.MODEL.CSFD.STYLE_AUG.PROB = 0.5         # probability per image
    
   #Ablations setting 
   #  cfg.MODEL.CSFD.USE_ROBUSTIFIER = True
   #  cfg.MODEL.CSFD.USE_GATE = True
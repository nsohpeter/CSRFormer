"""
Analyze which semantic classes get routed most by CASS in BDD100K.
Correlates per-query routing decisions with predicted class labels.
"""
import sys, torch, os, glob
sys.path.insert(0, '.')
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from mask2former import add_maskformer2_config
from mask2former.modeling.cass.cass_module import compute_attention_entropy

# Cityscapes 19-class names
CLASS_NAMES = [
    'road','sidewalk','building','wall','fence','pole',
    'traffic light','traffic sign','vegetation','terrain',
    'sky','person','rider','car','truck','bus',
    'train','motorcycle','bicycle'
]

cfg = get_cfg()
add_maskformer2_config(cfg)
cfg.set_new_allowed(True)
cfg.merge_from_file('custom_configs/training/cass_v2_swinb_90k.yaml')
cfg.MODEL.WEIGHTS = 'experiments/training_outputs/cass_v2_swinb_90k/model_final.pth'
cfg.MODEL.DEVICE  = 'cuda'
cfg.freeze()

model = build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
model.eval()

pixel_mean = torch.tensor([123.675, 116.28,  103.53]).view(3,1,1).cuda()
pixel_std  = torch.tensor([ 58.395,  57.12,   57.375]).view(3,1,1).cuda()

THRESHOLD = 4.5
N_IMAGES  = 200

# Storage: class_id -> [routed_count, total_count]
class_routing = {i: [0, 0] for i in range(19)}
class_routing[-1] = [0, 0]  # background/no-class

# Store routing decisions per layer per image
routing_decisions = {}  # layer_id -> list of routing masks [Q]

from mask2former.modeling.cass.cass_module import CASSLayer
orig_forward = CASSLayer.forward

def patched_forward(self, output, src, level_index, size_list,
                    attn_mask, pos, query_embed, current_iter=999999):
    from mask2former.modeling.cass.cass_module import (
        compute_soft_gate, select_blend_scale
    )
    out_curr, attn_w = self._run_cross_attn(
        output, src, level_index, size_list,
        attn_mask, pos, query_embed,
        ref_index=level_index, return_attn=True,
    )
    entropy    = compute_attention_entropy(attn_w)   # [B, Q]
    alpha_soft = compute_soft_gate(entropy, THRESHOLD, sigma=1.0, alpha_max=0.5)
    routing_mask = alpha_soft > 0.05                 # [B, Q]
    rate       = routing_mask.float().mean().item()
    mean_ent   = entropy.mean().item()

    # Log routing decisions into routing_decisions dict keyed by layer id
    lid = id(self)
    if lid not in routing_decisions:
        routing_decisions[lid] = []
    routing_decisions[lid].append(routing_mask[0].cpu())  # [Q]

    # Bidirectional blend
    coarser_idx, finer_idx = select_blend_scale(level_index, num_scales=3)
    blend_idx = coarser_idx if coarser_idx >= 0 else finer_idx
    if blend_idx >= 0 and alpha_soft.max() > 1e-4:
        out_blend = self._run_cross_attn(
            output, src, blend_idx, size_list,
            attn_mask, pos, query_embed, ref_index=level_index,
        )
        gate   = alpha_soft.permute(1, 0).unsqueeze(-1)
        output = (1.0 - gate) * out_curr + gate * out_blend
    else:
        output = out_curr

    return output, {'routing_rate': rate, 'mean_entropy': mean_ent,
                    'routing_mask': routing_mask}

CASSLayer.forward = patched_forward

# BDD images
paths = sorted(glob.glob(
    '/home/peter/research/datasets/bdd100k/images/val/*.jpg'
))[:N_IMAGES]

print(f"Analyzing {len(paths)} BDD images...")
print(f"Threshold τ = {THRESHOLD}\n")

with torch.no_grad():
    for idx, path in enumerate(paths):
        routing_decisions.clear()

        img = Image.open(path).convert('RGB').resize((1024, 512))
        x   = TF.to_tensor(img).unsqueeze(0).cuda() * 255.0
        x_n = (x - pixel_mean) / pixel_std

        feat    = model.backbone(x_n)
        model.sem_seg_head.predictor._current_iter = 999999
        outputs = model.sem_seg_head(feat)

        # Get predicted class per query [Q]
        pred_logits = outputs['pred_logits'][0]           # [Q, num_classes+1]
        pred_classes = pred_logits.argmax(dim=-1).cpu()   # [Q]

        # Aggregate routing per class
        # Combine routing decisions across all CASS layers
        # A query is "routed" if ANY layer routes it
        if routing_decisions:
            all_masks = torch.stack([
                masks[-1]  # last image's mask for this layer
                for masks in routing_decisions.values()
            ])  # [num_cass_layers, Q]
            any_routed = all_masks.any(dim=0)  # [Q] — routed in at least one layer

            for q in range(pred_classes.shape[0]):
                cls = pred_classes[q].item()
                if cls >= 19:
                    cls = -1  # background
                class_routing[cls][1] += 1      # total
                if any_routed[q].item():
                    class_routing[cls][0] += 1  # routed

        if (idx+1) % 50 == 0:
            print(f"  {idx+1}/{len(paths)} done")

# Report
print(f"\n{'='*60}")
print(f"BDD100K — Routing Rate per Predicted Class")
print(f"(queries routed in at least one CASS layer)")
print(f"{'='*60}")
print(f"{'Class':<20} {'Routed':>8} {'Total':>8} {'Rate':>8}")
print(f"{'-'*48}")

results = []
for cls_id in range(19):
    routed, total = class_routing[cls_id]
    if total > 0:
        rate = routed / total * 100
        results.append((rate, CLASS_NAMES[cls_id], routed, total))

results.sort(reverse=True)
for rate, name, routed, total in results:
    bar = '█' * int(rate/5)
    print(f"{name:<20} {routed:>8} {total:>8} {rate:>7.1f}%  {bar}")

# Background
bg_r, bg_t = class_routing[-1]
if bg_t > 0:
    print(f"{'(background)':<20} {bg_r:>8} {bg_t:>8} {bg_r/bg_t*100:>7.1f}%")

print(f"\nOverall routing rate: {sum(v[0] for v in class_routing.values()) / max(1, sum(v[1] for v in class_routing.values())) * 100:.1f}%")

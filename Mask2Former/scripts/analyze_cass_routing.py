"""
Analyze CASS routing behavior per domain.
Reports: routing rate per layer, entropy distribution per domain.
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

DOMAIN_PATHS = {
    'cityscapes': '/home/peter/research/datasets/cityscapes/leftImg8bit/val/*/*_leftImg8bit.png',
    'bdd100k':    '/home/peter/research/datasets/bdd100k/images/val/*.jpg',
    'gta5':       '/home/peter/research/datasets/gta5/images/*.png',
    'mapillary':  '/home/peter/research/datasets/mapillary_vistas/validation/images/*.jpg',
}

CASS_LAYERS   = [5, 6, 7, 8]
THRESHOLD     = 4.5
N_IMAGES      = 100

# Monkey-patch CASSLayer to log routing stats
routing_log = {}  # domain -> {layer -> [rates], entropy -> [vals]}

from mask2former.modeling.cass.cass_module import CASSLayer
orig_forward = CASSLayer.forward

def patched_forward(self, output, src, level_index, size_list,
                    attn_mask, pos, query_embed, current_iter=999999):
    from mask2former.modeling.cass.cass_module import (
        compute_soft_gate, select_blend_scale
    )
    # Pass 1: current scale cross-attention
    out_curr, attn_w = self._run_cross_attn(
        output, src, level_index, size_list,
        attn_mask, pos, query_embed,
        ref_index=level_index, return_attn=True,
    )
    entropy    = compute_attention_entropy(attn_w)   # [B, Q]
    alpha_soft = compute_soft_gate(entropy, THRESHOLD, self.sigma, self.alpha_max)
    rate       = (alpha_soft > 0.05).float().mean().item()
    mean_ent   = entropy.mean().item()

    if '_current_domain' in routing_log:
        dom = routing_log['_current_domain']
        if dom not in routing_log:
            routing_log[dom] = {}
        lid = id(self)
        if lid not in routing_log[dom]:
            routing_log[dom][lid] = {'rates': [], 'entropy': []}
        routing_log[dom][lid]['rates'].append(rate)
        routing_log[dom][lid]['entropy'].append(mean_ent)

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

    return output, {'routing_rate': rate, 'mean_entropy': mean_ent}

CASSLayer.forward = patched_forward

print(f"\n{'='*70}")
print(f"CASS Routing Analysis — threshold τ = {THRESHOLD}")
print(f"{'='*70}\n")

for domain, pattern in DOMAIN_PATHS.items():
    paths = sorted(glob.glob(pattern))[:N_IMAGES]
    if not paths:
        print(f"[{domain}] No images found at {pattern}")
        continue

    routing_log['_current_domain'] = domain
    routing_log[domain] = {}

    with torch.no_grad():
        for path in paths:
            img  = Image.open(path).convert('RGB').resize((1024, 512))
            x    = TF.to_tensor(img).unsqueeze(0).cuda() * 255.0
            x_n  = (x - pixel_mean) / pixel_std
            feat = model.backbone(x_n)
            model.sem_seg_head.predictor._current_iter = 999999
            model.sem_seg_head(feat)

    # Aggregate per CASS layer
    print(f"Domain: {domain.upper()}")
    print(f"  {'Layer':<8} {'Routing Rate':>14} {'Mean Entropy':>14} {'% above τ':>12}")
    print(f"  {'-'*52}")
    all_rates = []
    for lid, stats in routing_log[domain].items():
        rates = np.array(stats['rates'])
        ents  = np.array(stats['entropy'])
        print(f"  CASS-{lid%10:<3}   {rates.mean()*100:>12.1f}%   {ents.mean():>14.3f}   {(ents>THRESHOLD).mean()*100:>10.1f}%")
        all_rates.append(rates.mean())
    print(f"  Mean routing rate: {np.mean(all_rates)*100:.1f}%\n")

print("Done.")

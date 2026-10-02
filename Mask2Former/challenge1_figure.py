"""
Challenge 1 Figure: Layer-wise Domain Degradation Analysis
===========================================================
Shows that 82% of OOD degradation exists at layer 0 (pixel decoder output),
BEFORE the transformer decoder processes anything.
 
Uses the per-layer mIoU data already collected.
"""
 
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
# ---- Data from per-layer diagnosis ----
layers = list(range(10))
 
cityscapes = [72.685, 82.924, 82.808, 82.925, 82.816, 82.828, 82.853, 82.859, 82.791, 82.756]
fog        = [68.963, 74.359, 73.899, 73.708, 73.611, 74.071, 73.500, 73.750, 74.013, 73.815]
night      = [37.154, 38.404, 38.427, 38.451, 39.125, 38.884, 39.279, 39.047, 39.437, 39.297]
 
# Compute gaps
gap_fog   = [c - f for c, f in zip(cityscapes, fog)]
gap_night = [c - n for c, n in zip(cityscapes, night)]
 
output_dir = "experiments/figures/challenge_visuals"
os.makedirs(output_dir, exist_ok=True)
 
# ---- Figure: Layer-wise mIoU ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
 
# Left panel: mIoU per layer
for data, label, color, marker in [
    (cityscapes, 'Cityscapes (ID)', '#2166ac', 'o'),
    (fog, 'ACDC-Fog', '#fdae6b', 's'),
    (night, 'ACDC-Night', '#d62728', '^'),
]:
    ax1.plot(layers, data, color=color, marker=marker, markersize=7,
             linewidth=2, label=label, zorder=3)
 
# Annotate the layer 0 gap
ax1.annotate('',
    xy=(0, night[0]), xytext=(0, cityscapes[0]),
    arrowprops=dict(arrowstyle='<->', color='#d62728', lw=2))
ax1.text(-0.7, (cityscapes[0] + night[0]) / 2,
         f'35.5 mIoU gap\nat Layer 0\n(82% of total)',
         fontsize=9, color='#d62728', fontweight='bold',
         ha='center', va='center',
         bbox=dict(facecolor='#fff0f0', edgecolor='#d62728',
                   boxstyle='round,pad=0.4', alpha=0.9))
 
# Annotate layer 1 big jump for Cityscapes
ax1.annotate('+10.2',
    xy=(1, cityscapes[1]), xytext=(1.8, 78),
    fontsize=8, color='#2166ac',
    arrowprops=dict(arrowstyle='->', color='#2166ac', lw=1.2),
    bbox=dict(facecolor='white', edgecolor='#2166ac',
              boxstyle='round,pad=0.3', alpha=0.8))
 
# Shade the "before decoder" vs "decoder" regions
ax1.axvspan(-0.5, 0.5, alpha=0.08, color='#d62728', zorder=0)
ax1.text(0, 34, 'Pixel Decoder\nOutput', ha='center', fontsize=8,
         color='#d62728', fontstyle='italic', alpha=0.7)
ax1.axvspan(0.5, 9.5, alpha=0.04, color='#2166ac', zorder=0)
ax1.text(5, 34, 'Transformer Decoder Layers', ha='center', fontsize=8,
         color='#2166ac', fontstyle='italic', alpha=0.7)
 
ax1.set_xlabel('Decoder Layer', fontsize=11)
ax1.set_ylabel('mIoU', fontsize=11)
ax1.set_title('Layer-wise Prediction Quality', fontsize=13, fontweight='bold')
ax1.set_xticks(layers)
ax1.set_xticklabels([f'L{i}' for i in layers], fontsize=9)
ax1.set_ylim(30, 88)
ax1.legend(fontsize=10, loc='lower right')
ax1.grid(axis='y', alpha=0.3)
 
# Right panel: ID-OOD gap per layer
ax2.fill_between(layers, gap_night, alpha=0.15, color='#d62728')
ax2.plot(layers, gap_night, color='#d62728', marker='^', markersize=7,
         linewidth=2, label='Cityscapes → Night gap', zorder=3)
ax2.fill_between(layers, gap_fog, alpha=0.15, color='#fdae6b')
ax2.plot(layers, gap_fog, color='#fdae6b', marker='s', markersize=7,
         linewidth=2, label='Cityscapes → Fog gap', zorder=3)
 
# Annotate gap jump from layer 0 to layer 1
ax2.annotate('Gap widens by\nomission, not\nby degradation',
    xy=(1, gap_night[1]), xytext=(3, 48),
    fontsize=9, color='#333333',
    arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5),
    bbox=dict(facecolor='#f5f5f5', edgecolor='#999999',
              boxstyle='round,pad=0.4', alpha=0.9))
 
# Annotate layer 0
ax2.axvspan(-0.5, 0.5, alpha=0.08, color='#d62728', zorder=0)
 
ax2.set_xlabel('Decoder Layer', fontsize=11)
ax2.set_ylabel('ID - OOD mIoU Gap', fontsize=11)
ax2.set_title('Domain Shift Gap Across Layers', fontsize=13, fontweight='bold')
ax2.set_xticks(layers)
ax2.set_xticklabels([f'L{i}' for i in layers], fontsize=9)
ax2.legend(fontsize=10, loc='center right')
ax2.grid(axis='y', alpha=0.3)
 
fig.suptitle('Challenge 1: Where Does Domain Degradation Originate?',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
 
path = os.path.join(output_dir, 'challenge1_layer_degradation.png')
plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {path}")
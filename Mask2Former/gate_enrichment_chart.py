"""
Gate boundary-enrichment bar chart -- quantitative companion to Figure B.

Plots Enrichment vs GT-boundary (the more rigorous of the two enrichment
metrics computed in figureB_gate_fusion.py -- tested against real
class-transition boundaries, not raw pixel contrast) across domains,
with a dashed chance-level reference at 1.0x.

Data pasted directly from a completed figureB_gate_fusion.py run
(--num-images 30). Update DATA below if you rerun with different
domains, more images, or a different checkpoint.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# domain -> (Enrich/GTbound, n images)
DATA = {
    'Cityscapes': (2.46, 30),
    'BDD100K':    (2.78, 30),
    'GTA5':       (2.32, 30),
    'ACDC-Night': (1.89, 30),
}

DOMAIN_COLORS = {
    'Cityscapes': '#6baed6',
    'BDD100K':    '#fdae6b',
    'GTA5':       '#74c476',
    'ACDC-Night': '#756bb1',
}

labels = list(DATA.keys())
values = [DATA[k][0] for k in labels]
colors = [DOMAIN_COLORS[k] for k in labels]

fig, ax = plt.subplots(figsize=(6.5, 4.5))

x = np.arange(len(labels))
bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.6, width=0.6)

# chance-level reference line
ax.axhline(1.0, color='#333333', linestyle='--', linewidth=1.2, zorder=1)
ax.text(len(labels) - 0.45, 1.05, 'chance level (1.0×)', fontsize=9,
        color='#333333', ha='right', va='bottom')

# value labels above each bar
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.06, f'{v:.2f}×',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Gate enrichment vs. GT boundary\n(× chance)', fontsize=11)
ax.set_title('Where the gate focuses: enrichment against\nground-truth semantic boundaries',
              fontsize=13, fontweight='bold', pad=12)
ax.set_ylim(0, max(values) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.25, zorder=0)

plt.tight_layout()
import os
os.makedirs('experiments/figures/gate_enrichment', exist_ok=True)
plt.savefig('experiments/figures/gate_enrichment/gate_boundary_enrichment.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('experiments/figures/gate_enrichment/gate_boundary_enrichment.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved PNG and PDF to experiments/figures/gate_enrichment/")
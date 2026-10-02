"""
visualize_stability.py
Generate publication-quality figures from the stability analysis results.
 
Produces:
    1. Layer-wise CA JSD profile  (main figure — 3 subplots, one per target domain)
       Shows how cross-attention instability evolves across the 9 decoder layers
       for Vanilla M2F vs CMFormer (+ your stability module once trained).
 
    2. MCAJ comparison bar chart  (summary figure)
       One grouped bar per target domain, one bar per model.
 
Place at:  ~/research/visualize_stability.py
Run from:  ~/research/
 
Output:
    shared_experiments/stability_analysis/results/figures/layer_profile.pdf
    shared_experiments/stability_analysis/results/figures/layer_profile.png
    shared_experiments/stability_analysis/results/figures/mcaj_comparison.pdf
    shared_experiments/stability_analysis/results/figures/mcaj_comparison.png
"""
 
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
 
matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.linewidth":   0.8,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "xtick.direction":  "out",
    "ytick.direction":  "out",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "legend.frameon":   False,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})
 
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESEARCH_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR   = os.path.join(RESEARCH_DIR, "shared_experiments", "stability_analysis", "results")
FIGURES_DIR   = os.path.join(RESULTS_DIR, "figures")
PROFILE_CSV   = os.path.join(RESULTS_DIR, "layer_profile.csv")
TABLE_CSV     = os.path.join(RESULTS_DIR, "comparison_table.csv")
 
# ---------------------------------------------------------------------------
# Style config — add your model here when ready
# ---------------------------------------------------------------------------
MODEL_STYLE = {
    "m2f": {
        "label":  "Vanilla M2F",
        "color":  "#2166ac",
        "ls":     "-",
        "marker": "o",
        "lw":     1.8,
        "ms":     5,
        "zorder": 3,
    },
    "cmformer": {
        "label":  "CMFormer",
        "color":  "#d6604d",
        "ls":     "--",
        "marker": "s",
        "lw":     1.8,
        "ms":     5,
        "zorder": 3,
    },
    "ours": {
        "label":  "Ours (stability module)",
        "color":  "#1a9641",
        "ls":     "-.",
        "marker": "^",
        "lw":     2.0,
        "ms":     6,
        "zorder": 4,
    },
     "csfd": {
        "label":  "CSFD (ours)",
        "color":  "#6a3d9a",
        "ls":     "-.",
        "marker": "D",
        "lw":     2.0,
        "ms":     6,
        "zorder": 5,
    },
}
 
TARGET_LABELS = {
    "bdd":       "Cityscapes → BDD100K",
    "gta5":      "Cityscapes → GTA5",
    "mapillary": "Cityscapes → Mapillary",
}
 
# Scale group background shading (layers 0-2, 3-5, 6-8)
SCALE_GROUPS = [
    (0, 2, "#f7f7f7", "Pass 1"),
    (3, 5, "#e8e8e8", "Pass 2"),
    (6, 8, "#f7f7f7", "Pass 3"),
]
 
NUM_LAYERS = 9
LAYER_TICKS = list(range(NUM_LAYERS))
LAYER_LABELS = [f"L{i}\n(S{i%3})" for i in range(NUM_LAYERS)]
 
 
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
 
def load_profile() -> pd.DataFrame:
    if not os.path.exists(PROFILE_CSV):
        raise FileNotFoundError(f"Layer profile CSV not found: {PROFILE_CSV}\nRun compare_models.py first.")
    return pd.read_csv(PROFILE_CSV)
 
 
def load_table() -> pd.DataFrame:
    if not os.path.exists(TABLE_CSV):
        raise FileNotFoundError(f"Comparison table CSV not found: {TABLE_CSV}\nRun compare_models.py first.")
    return pd.read_csv(TABLE_CSV)
 
 
def get_profile_array(df: pd.DataFrame, model_key: str, target_key: str) -> np.ndarray:
    """Extract [NUM_LAYERS] CA JSD array for one model/target pair."""
    mask = (df["model_key"] == model_key) & (df["target_key"] == target_key)
    sub  = df[mask].sort_values("layer")
    if sub.empty:
        return None
    return sub["ca_jsd"].values
 
 
# ---------------------------------------------------------------------------
# Figure 1 — Layer-wise CA JSD profile
# ---------------------------------------------------------------------------
 
def plot_layer_profile(df: pd.DataFrame):
    targets     = list(TARGET_LABELS.keys())
    model_keys  = [k for k in MODEL_STYLE if k in df["model_key"].unique()]
    n_targets   = len(targets)
 
    fig, axes = plt.subplots(1, n_targets, figsize=(5.5 * n_targets, 4.5), sharey=True)
    if n_targets == 1:
        axes = [axes]
 
    for ax_idx, (target_key, ax) in enumerate(zip(targets, axes)):
 
        # Scale group shading
        for lo, hi, color, _ in SCALE_GROUPS:
            ax.axvspan(lo - 0.5, hi + 0.5, color=color, alpha=0.6, zorder=0)
 
        # Vertical separators between passes
        ax.axvline(2.5, color="#bbbbbb", lw=0.8, ls=":", zorder=1)
        ax.axvline(5.5, color="#bbbbbb", lw=0.8, ls=":", zorder=1)
 
        # Plot each model
        for model_key in model_keys:
            profile = get_profile_array(df, model_key, target_key)
            if profile is None:
                continue
            s = MODEL_STYLE[model_key]
            ax.plot(
                LAYER_TICKS, profile,
                color=s["color"], ls=s["ls"], lw=s["lw"],
                marker=s["marker"], markersize=s["ms"],
                zorder=s["zorder"], label=s["label"],
            )
 
        # Axis formatting
        ax.set_title(TARGET_LABELS[target_key], fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks(LAYER_TICKS)
        ax.set_xticklabels(LAYER_LABELS, fontsize=9)
        ax.set_xlim(-0.6, NUM_LAYERS - 0.4)
        ax.set_ylim(0.25, 0.95)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
        ax.grid(axis="y", lw=0.5, alpha=0.4, zorder=0)
 
        if ax_idx == 0:
            ax.set_ylabel("Mean CA JSD  (lower = more stable)", fontsize=11)
 
        # Pass labels at top
        for lo, hi, _, pass_label in SCALE_GROUPS:
            mid = (lo + hi) / 2
            ax.text(mid, 0.935, pass_label, ha="center", va="bottom",
                    fontsize=8, color="#555555",
                    transform=ax.get_xaxis_transform())
 
    # Shared legend
    handles = [
        Line2D([0], [0],
               color=MODEL_STYLE[mk]["color"],
               ls=MODEL_STYLE[mk]["ls"],
               lw=MODEL_STYLE[mk]["lw"],
               marker=MODEL_STYLE[mk]["marker"],
               markersize=MODEL_STYLE[mk]["ms"],
               label=MODEL_STYLE[mk]["label"])
        for mk in model_keys
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(model_keys),
        fontsize=11,
        bbox_to_anchor=(0.5, -0.06),
        columnspacing=2.0,
    )
 
    fig.suptitle(
        "Cross-Attention Instability Across Decoder Layers",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    return fig
 
 
# ---------------------------------------------------------------------------
# Figure 2 — MCAJ comparison bar chart
# ---------------------------------------------------------------------------
 
def plot_mcaj_bars(df: pd.DataFrame):
    model_keys  = [k for k in MODEL_STYLE if k in df["model_key"].unique()]
    target_keys = list(TARGET_LABELS.keys())
 
    n_models  = len(model_keys)
    n_targets = len(target_keys)
    x         = np.arange(n_targets)
    width     = 0.22
    offsets   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width
 
    fig, ax = plt.subplots(figsize=(7, 4.5))
 
    for m_idx, model_key in enumerate(model_keys):
        s    = MODEL_STYLE[model_key]
        vals = []
        for target_key in target_keys:
            row = df[(df["model_key"] == model_key) & (df["target_key"] == target_key)]
            vals.append(float(row["MCAJ"].values[0]) if not row.empty else 0.0)
 
        bars = ax.bar(
            x + offsets[m_idx], vals,
            width=width * 0.9,
            color=s["color"], alpha=0.85,
            label=s["label"], zorder=3,
        )
 
        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center", va="bottom",
                fontsize=8, color="#333333",
            )
 
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[t] for t in target_keys], fontsize=10)
    ax.set_ylabel("MCAJ  (lower = more stable)", fontsize=11)
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.grid(axis="y", lw=0.5, alpha=0.4, zorder=0)
    ax.set_title(
        "Mean Cross-Attention JSD (MCAJ) by Domain",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=10, loc="upper left")
    plt.tight_layout()
    return fig
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
 
    profile_df = load_profile()
    table_df   = load_table()
 
    print(f"\nModels found in data : {sorted(profile_df['model_key'].unique())}")
    print(f"Targets found in data: {sorted(profile_df['target_key'].unique())}")
 
    # Figure 1 — layer profile
    print("\nGenerating layer profile figure ...")
    fig1 = plot_layer_profile(profile_df)
    for ext in ["pdf", "png"]:
        path = os.path.join(FIGURES_DIR, f"layer_profile.{ext}")
        fig1.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig1)
 
    # Figure 2 — MCAJ bars
    print("\nGenerating MCAJ comparison figure ...")
    fig2 = plot_mcaj_bars(table_df)
    for ext in ["pdf", "png"]:
        path = os.path.join(FIGURES_DIR, f"mcaj_comparison.{ext}")
        fig2.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig2)
 
    print("\nDone.\n")
 
 
if __name__ == "__main__":
    main()
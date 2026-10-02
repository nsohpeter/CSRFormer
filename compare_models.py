"""
compare_models.py
Reads all cross-domain JSD CSVs and mIoU evaluation logs, then produces:
 
    1. Unified comparison table  (mIoU + MCAJ per model per target domain)
       → shared_experiments/stability_analysis/results/comparison_table.csv
 
    2. Layer-wise CA JSD profile  (per layer mean JSD per model per target)
       → shared_experiments/stability_analysis/results/layer_profile.csv
 
    3. Clean stdout summary report
 
Place at:  ~/research/compare_models.py
Run from:  ~/research/
 
When your stability module is trained:
    - Add its 3 JSD CSVs to jsd_csvs/ named ours_cs_to_{bdd,gta5,mapillary}.csv
    - Add its 3 mIoU log paths to MIOU_LOG_PATHS below
    - Rerun — the table gains a third row automatically
"""
 
import os
import re
import numpy as np
import pandas as pd
 
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
JSD_CSV_DIR  = os.path.join(RESEARCH_DIR, "shared_experiments", "stability_analysis", "results", "jsd_csvs")
RESULTS_DIR  = os.path.join(RESEARCH_DIR, "shared_experiments", "stability_analysis", "results")
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS = {
    "m2f":      "Vanilla M2F",
    "cmformer": "CMFormer",
    "csfd":     "CSFD (ours)"
}
 
TARGETS = {
    "bdd":      "→BDD100K",
    "gta5":     "→GTA5",
    "mapillary": "→Mapillary",
}
 
NUM_LAYERS = 9
 
# mIoU log paths — add "ours" entry when your module is trained
MIOU_LOG_PATHS = {
    "m2f": {
        "bdd":       "Mask2Former/experiments/evaluation_results/miou_scores/bdd100k_sem_seg_val.log",
        "gta5":      "Mask2Former/experiments/evaluation_results/miou_scores/gta5_sem_seg_val.log",
        "mapillary": "Mask2Former/experiments/evaluation_results/miou_scores/mapillary_vistas_sem_seg_val.log",
    },
    "cmformer": {
        "bdd":       "CMFormer/experiments/evaluation_results/miou_scores/bdd100k_sem_seg_val_final.log",
        "gta5":      "CMFormer/experiments/evaluation_results/miou_scores/gta5_sem_seg_val_final.log",
        "mapillary": "CMFormer/experiments/evaluation_results/miou_scores/mapillary_vistas_sem_seg_val_final.log",
    },
     "csfd": {
        "bdd":       "Mask2Former/experiments/stability_evaluation_results/miou_scores/bdd100k_csfd.log",
        "gta5":      "Mask2Former/experiments/stability_evaluation_results/miou_scores/gta5_csfd.log",
        "mapillary": "Mask2Former/experiments/stability_evaluation_results/miou_scores/mapillary_csfd.log",
    },
    
}
 
 
# ---------------------------------------------------------------------------
# mIoU parser
# ---------------------------------------------------------------------------
 
def parse_miou_from_log(log_path: str) -> float:
    """
    Extract mIoU from a Detectron2 evaluation log.
    Looks for the copypaste line: 'copypaste: 59.71,86.52,...'
    Returns None if file not found or pattern not matched.
    """
    full_path = os.path.join(RESEARCH_DIR, log_path)
    if not os.path.exists(full_path):
        return None
    with open(full_path, "r") as f:
        for line in f:
            if "copypaste:" in line:
                # Skip the header line (contains 'mIoU,fwIoU,...')
                if "mIoU,fwIoU" in line:
                    continue
                # Extract numeric part after 'copypaste: '
                match = re.search(r"copypaste:\s*([\d.]+),", line)
                if match:
                    return float(match.group(1))
    return None
 
 
# ---------------------------------------------------------------------------
# JSD helpers
# ---------------------------------------------------------------------------
 
def load_jsd_csv(model_key: str, target_key: str) -> pd.DataFrame:
    fname = f"{model_key}_cs_to_{target_key}.csv"
    path  = os.path.join(JSD_CSV_DIR, fname)
    return pd.read_csv(path) if os.path.exists(path) else None
 
 
def compute_mcaj(df: pd.DataFrame) -> float:
    head_cols = [
        f"ca_jsd_layer{i}_head{h}"
        for i in range(NUM_LAYERS) for h in range(8)
    ]
    available = [c for c in head_cols if c in df.columns]
    if not available:
        available = [f"ca_jsd_layer{i}_mean" for i in range(NUM_LAYERS)
                     if f"ca_jsd_layer{i}_mean" in df.columns]
    return float(df[available].values.mean())
 
 
def compute_layer_profile(df: pd.DataFrame) -> np.ndarray:
    profile = np.zeros(NUM_LAYERS)
    for i in range(NUM_LAYERS):
        head_cols  = [f"ca_jsd_layer{i}_head{h}" for h in range(8)]
        available  = [c for c in head_cols if c in df.columns]
        if available:
            profile[i] = float(df[available].values.mean())
        else:
            profile[i] = float(df[f"ca_jsd_layer{i}_mean"].mean())
    return profile
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
 
    table_rows   = []
    profile_rows = []
 
    for model_key, model_name in MODELS.items():
        for target_key, target_name in TARGETS.items():
 
            # --- mIoU ---
            log_paths   = MIOU_LOG_PATHS.get(model_key, {})
            log_path    = log_paths.get(target_key)
            miou        = parse_miou_from_log(log_path) if log_path else None
 
            # --- MCAJ ---
            df   = load_jsd_csv(model_key, target_key)
            mcaj = compute_mcaj(df) if df is not None else None
 
            table_rows.append({
                "model":      model_name,
                "model_key":  model_key,
                "target":     target_name,
                "target_key": target_key,
                "mIoU":       round(miou, 4) if miou is not None else None,
                "MCAJ":       round(mcaj, 4) if mcaj is not None else None,
            })
 
            # --- layer profile ---
            if df is not None:
                profile = compute_layer_profile(df)
                for layer_idx, jsd_val in enumerate(profile):
                    profile_rows.append({
                        "model":       model_name,
                        "model_key":   model_key,
                        "target":      target_name,
                        "target_key":  target_key,
                        "layer":       layer_idx,
                        "scale_group": layer_idx % 3,
                        "ca_jsd":      round(float(jsd_val), 4),
                    })
 
    table_df   = pd.DataFrame(table_rows)
    profile_df = pd.DataFrame(profile_rows)
 
    table_df.to_csv(os.path.join(RESULTS_DIR, "comparison_table.csv"),   index=False)
    profile_df.to_csv(os.path.join(RESULTS_DIR, "layer_profile.csv"), index=False)
 
    # ------------------------------------------------------------------ #
    # Stdout report
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 78)
    print("  ATTENTION STABILITY + SEGMENTATION PERFORMANCE — UNIFIED REPORT")
    print("=" * 78)
    print("  MCAJ: Mean Cross-Attention JSD  |  lower = more stable")
    print("  mIoU: mean Intersection over Union (%)  |  higher = better")
    print("-" * 78)
 
    cw = 11   # column width
 
    # Header
    hdr  = f"{'Model':<26}"
    hdr += f"{'mIoU→BDD':>{cw}}{'mIoU→GTA5':>{cw}}{'mIoU→Map':>{cw}}{'Mean mIoU':>{cw}}"
    hdr += f"{'MCAJ→BDD':>{cw}}{'MCAJ→GTA5':>{cw}}{'MCAJ→Map':>{cw}}{'Mean MCAJ':>{cw}}"
    print(f"\n{hdr}")
    print("-" * 78)
 
    for model_key, model_name in MODELS.items():
        rows = table_df[table_df["model_key"] == model_key]
        if rows.empty:
            continue
 
        miou_vals, mcaj_vals = [], []
        miou_strs, mcaj_strs = [], []
 
        for target_key in TARGETS:
            row = rows[rows["target_key"] == target_key]
            if row.empty:
                miou_strs.append("—")
                mcaj_strs.append("—")
                continue
 
            m = row["mIoU"].values[0]
            c = row["MCAJ"].values[0]
 
            if pd.notna(m):
                miou_strs.append(f"{m:.2f}")
                miou_vals.append(m)
            else:
                miou_strs.append("—")
 
            if pd.notna(c):
                mcaj_strs.append(f"{c:.4f}")
                mcaj_vals.append(c)
            else:
                mcaj_strs.append("—")
 
        mean_miou = f"{np.mean(miou_vals):.2f}" if miou_vals else "—"
        mean_mcaj = f"{np.mean(mcaj_vals):.4f}" if mcaj_vals else "—"
 
        row_str  = f"{model_name:<26}"
        row_str += "".join(f"{v:>{cw}}" for v in miou_strs)
        row_str += f"{mean_miou:>{cw}}"
        row_str += "".join(f"{v:>{cw}}" for v in mcaj_strs)
        row_str += f"{mean_mcaj:>{cw}}"
        print(row_str)
 
    print("-" * 78)
 
    # Layer-wise breakdown
    print("\n  Layer-wise CA JSD (mean over heads + 500 images)\n")
    combos       = [(mk, tk) for mk in ["m2f", "cmformer", "csfd"] for tk in ["bdd", "gta5", "mapillary"]]
    combo_labels = [f"{mk[:3]}→{tk[:3]}" for mk, tk in combos]
 
    print(f"  {'Layer':<8}{'Scale':<8}" + "".join(f"{lb:>10}" for lb in combo_labels))
    print("  " + "-" * (16 + 10 * len(combos)))
 
    for layer_idx in range(NUM_LAYERS):
        scale = f"S{layer_idx % 3}"
        vals  = []
        for model_key, target_key in combos:
            row = profile_df[
                (profile_df["model_key"]  == model_key) &
                (profile_df["target_key"] == target_key) &
                (profile_df["layer"]      == layer_idx)
            ]
            vals.append(f"{row['ca_jsd'].values[0]:.4f}" if not row.empty else "—")
        print(f"  {layer_idx:<8}{scale:<8}" + "".join(f"{v:>10}" for v in vals))
 
    print("\n" + "=" * 78)
    print(f"  Saved: {os.path.join(RESULTS_DIR, 'comparison_table.csv')}")
    print(f"  Saved: {os.path.join(RESULTS_DIR, 'layer_profile.csv')}")
    print("=" * 78 + "\n")
 
 
if __name__ == "__main__":
    main()
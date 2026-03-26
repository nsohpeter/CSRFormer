import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
RUNS = {
    "Baseline (pretrained_v2)": "experiments/training_outputs/baseline_pretrained_v2/metrics.json",
    "Stability Module":         "experiments/training_outputs/stability_module/metrics.json",
}

SMOOTH_WINDOW = 200   # iterations, set to 1 to disable
OUTPUT_DIR    = "experiments/phase4_stability_module/reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ──────────────────────────────────────────────────────────────────────────────

def load_metrics(path):
    train, val = [], []
    with open(path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if "total_loss" in entry:
                train.append(entry)
            elif "sem_seg/IoU" in entry:
                val.append(entry)
    return train, val

def smooth(values, window):
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")

def get(records, key):
    return (
        [r["iteration"] for r in records if key in r],
        [r[key]         for r in records if key in r],
    )

# ─── LOAD ─────────────────────────────────────────────────────────────────────
all_train, all_val = {}, {}
for name, path in RUNS.items():
    if not os.path.exists(path):
        print(f"  [skip] {name} — file not found: {path}")
        continue
    t, v = load_metrics(path)
    all_train[name] = t
    all_val[name]   = v
    print(f"  [ok] {name}: {len(t)} train entries, {len(v)} val entries")

colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

# ─── PLOT 1: Total Loss ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
for i, (name, records) in enumerate(all_train.items()):
    iters, vals = get(records, "total_loss")
    ax.plot(iters, smooth(vals, SMOOTH_WINDOW), label=name, color=colors[i], linewidth=1.5)
ax.set_xlabel("Iteration")
ax.set_ylabel("Total Loss")
ax.set_title("Training — Total Loss")
ax.legend()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/loss_total.png", dpi=150)
plt.savefig(f"{OUTPUT_DIR}/loss_total.pdf")
plt.close()
print("  saved: loss_total")

# ─── PLOT 2: Loss Components ──────────────────────────────────────────────────
components = {"loss_ce": "CE Loss", "loss_mask": "Mask Loss", "loss_dice": "Dice Loss"}
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (key, title) in zip(axes, components.items()):
    for i, (name, records) in enumerate(all_train.items()):
        iters, vals = get(records, key)
        ax.plot(iters, smooth(vals, SMOOTH_WINDOW), label=name, color=colors[i], linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
plt.suptitle("Training — Loss Components", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/loss_components.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUTPUT_DIR}/loss_components.pdf", bbox_inches="tight")
plt.close()
print("  saved: loss_components")

# ─── PLOT 3: Learning Rate ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 3))
for i, (name, records) in enumerate(all_train.items()):
    iters, vals = get(records, "lr")
    ax.plot(iters, vals, label=name, color=colors[i], linewidth=1.5)
ax.set_xlabel("Iteration")
ax.set_ylabel("Learning Rate")
ax.set_title("Learning Rate Schedule")
ax.legend()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/lr_schedule.png", dpi=150)
plt.savefig(f"{OUTPUT_DIR}/lr_schedule.pdf")
plt.close()
print("  saved: lr_schedule")

# ─── PLOT 4: Validation mIoU ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
has_val = False
for i, (name, records) in enumerate(all_val.items()):
    if not records:
        print(f"  [skip] no val mIoU entries for: {name}")
        continue
    iters, vals = get(records, "sem_seg/IoU")
    ax.plot(iters, vals, "o-", label=name, color=colors[i], linewidth=1.5, markersize=4)
    has_val = True
if has_val:
    ax.set_xlabel("Iteration")
    ax.set_ylabel("mIoU (%)")
    ax.set_title("Validation — mIoU (Cityscapes)")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/val_miou.png", dpi=150)
    plt.savefig(f"{OUTPUT_DIR}/val_miou.pdf")
    plt.close()
    print("  saved: val_miou")

print(f"\n✓ All plots saved to {OUTPUT_DIR}/")

"""
One-time check: confirm the trainId class order MetadataCatalog reports
for the Cityscapes dataset used in training, so the CLASS_FREQUENCIES
array for logit adjustment is indexed correctly (index i must correspond
to trainId i, or the wrong prior gets applied to the wrong class).

Run once from the Mask2Former repo root:
    python check_class_order.py
"""
import sys

# Import the project's dataset registration so custom datasets
# (cityscapes_fine_sem_seg_train, bdd100k_sem_seg_val, etc.) actually
# get registered with MetadataCatalog before we query it. Mask2Former's
# train_net.py normally does this via its own imports -- mirror that here.
try:
    import mask2former  # noqa: F401  (registers datasets as a side effect, same as train_net.py)
except ImportError as e:
    print(f"WARNING: could not import mask2former package ({e}). "
          f"If dataset registration happens elsewhere (e.g. a custom "
          f"register_datasets.py), import that instead.")

from detectron2.data import MetadataCatalog, DatasetCatalog

# Candidate dataset names to check -- adjust/add if yours differ.
# We check several because different Mask2Former configs sometimes
# register the same underlying data under different names.
CANDIDATE_NAMES = [
    "cityscapes_fine_sem_seg_train",
    "cityscapes_fine_sem_seg_val",
]

print("Registered dataset names containing 'cityscapes':")
for name in sorted(DatasetCatalog.list()):
    if "cityscapes" in name.lower():
        print(f"  {name}")
print()

found_any = False
for name in CANDIDATE_NAMES:
    try:
        meta = MetadataCatalog.get(name)
    except KeyError:
        print(f"'{name}': not registered, skipping")
        continue

    found_any = True
    classes = getattr(meta, "stuff_classes", None)
    if classes is None:
        print(f"'{name}': registered but has no stuff_classes attribute")
        continue

    print(f"'{name}' stuff_classes (index = trainId):")
    for i, c in enumerate(classes):
        print(f"  {i:2d}: {c}")
    print()

if not found_any:
    print("None of the candidate names were registered. Run this instead "
          "to see everything available, then re-run this script with the "
          "correct name added to CANDIDATE_NAMES:")
    print("  from detectron2.data import DatasetCatalog")
    print("  print(sorted(DatasetCatalog.list()))")
    sys.exit(1)
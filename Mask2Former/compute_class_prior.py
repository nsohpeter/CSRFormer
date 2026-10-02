"""
Compute class prior for logit adjustment
========================================
Logit adjustment needs the per-class prior the MASK CLASSIFIER is trained on.
In Mask2Former semantic training each GT class present in an image becomes one
classification target (one binary mask), so the right prior is the per-class
GT-MASK frequency across the training set (≈ how many (image, class) targets
exist per class) — NOT pixel frequency (which over-weights road/building).
 
Saves cityscapes_train_class_prior.pt = {prior: [K], log_prior: [K], counts: [K],
classes: [...]}. The criterion loads log_prior and adds tau*log_prior to logits.
 
Run:  python compute_class_prior.py
"""
 
import numpy as np
import torch
from PIL import Image
 
TRAIN_DATASET = "cityscapes_fine_sem_seg_train"
OUT = "cityscapes_train_class_prior.pt"
CONFIG_FILE = "custom_configs/training/csfd_v2_swinb_90k.yaml"   # any cfg that registers datasets
 
 
def main():
    import train_net
    from detectron2.engine import default_argument_parser
    from detectron2.data import DatasetCatalog, MetadataCatalog
    args = default_argument_parser().parse_args(["--config-file", CONFIG_FILE])
    train_net.setup(args)
 
    dicts = DatasetCatalog.get(TRAIN_DATASET)
    meta = MetadataCatalog.get(TRAIN_DATASET)
    classes = meta.stuff_classes; K = len(classes)
    ignore = getattr(meta, "ignore_label", 255)
 
    counts = np.zeros(K, dtype=np.int64)   # # images containing each class = # mask targets
    for i, d in enumerate(dicts):
        gt = np.array(Image.open(d["sem_seg_file_name"]), dtype=np.int64)
        present = np.unique(gt[gt != ignore])
        counts[present[present < K]] += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(dicts)}")
 
    prior = counts / counts.sum()
    prior = np.clip(prior, 1e-6, None)     # avoid log(0) for any absent class
    log_prior = np.log(prior)
 
    torch.save({
        "prior": torch.tensor(prior, dtype=torch.float32),
        "log_prior": torch.tensor(log_prior, dtype=torch.float32),
        "counts": torch.tensor(counts),
        "classes": list(classes),
    }, OUT)
 
    print(f"\nsaved {OUT}")
    print(f"{'class':16s}{'count':>8s}{'prior%':>9s}{'log_prior':>11s}")
    for i in np.argsort(counts):
        print(f"{classes[i]:16s}{counts[i]:8d}{prior[i]*100:9.2f}{log_prior[i]:11.3f}")
    print("\nMost-negative log_prior = rarest = gets the biggest boost under logit adjustment.")
    print("Confirm the confusion victims (wall/fence/pole/truck/bus) sit toward the rare end.")
 
 
if __name__ == "__main__":
    main()
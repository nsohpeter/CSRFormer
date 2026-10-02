"""
BDD class-frequency + present-class mIoU
========================================
Tests whether part of BDD's "low mean" is just rare-class averaging (e.g. train
is near-absent in US dashcam scenes, so IoU-train~0 drags the 19-class mean down
regardless of the model).
 
Prints, from GROUND TRUTH only (no model needed for the frequency part; the model
is only used to also report present-class mIoU):
  1. per class: % of val images containing it, and % of labelled pixels it covers
  2. classes that are effectively absent (image-freq below a threshold)
  3. standard 19-class mIoU vs mIoU over PRESENT classes only
 
Compares BDD against Cityscapes so "rare in BDD" is relative to the source.
 
Point MODEL at vanilla (rare-class averaging is a property of the data + a normal
model, not of CSFD).  Run:  python diagnose_bdd_class_freq.py
"""
 
import numpy as np
import torch
from PIL import Image
 
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_FILE = "custom_configs/training/vanilla_mask2former_swinb_90k.yaml"
WEIGHTS     = "experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth"
 
DATASETS = {
    "cityscapes": "cityscapes_fine_sem_seg_val",
    "bdd":        "bdd100k_sem_seg_val",
}
ABSENT_IMG_FREQ = 5.0   # class present in < this % of images = "effectively absent"
 
 
def gt_index(dataset_name):
    from detectron2.data import DatasetCatalog, MetadataCatalog
    dicts = DatasetCatalog.get(dataset_name)
    meta = MetadataCatalog.get(dataset_name)
    fmap = {d["file_name"]: d["sem_seg_file_name"] for d in dicts}
    return fmap, meta.stuff_classes, getattr(meta, "ignore_label", 255)
 
 
def class_frequency(fmap, K, ignore):
    """From GT only: per-class image-frequency (%) and pixel-share (%)."""
    n_img = len(fmap)
    img_count = np.zeros(K, dtype=np.int64)
    pix_count = np.zeros(K, dtype=np.int64)
    tot_pix = 0
    for gt_path in fmap.values():
        gt = np.array(Image.open(gt_path), dtype=np.int64)
        valid = gt[gt != ignore]
        tot_pix += valid.size
        present = np.unique(valid)
        binc = np.bincount(valid, minlength=K)[:K]
        pix_count += binc
        img_count[present[present < K]] += 1
    return (img_count / max(n_img, 1) * 100.0,
            pix_count / max(tot_pix, 1) * 100.0)
 
 
@torch.no_grad()
def per_class_iou(cfg, dataset_name, fmap, K, ignore):
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import build_detection_test_loader
    model = build_model(cfg).eval().to(DEVICE)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    conf = np.zeros((K + 1, K + 1), dtype=np.int64)
    for batch in build_detection_test_loader(cfg, dataset_name):
        for inp, out in zip(batch, model(batch)):
            pred = out["sem_seg"].argmax(0).to("cpu").numpy().astype(np.int64)
            gt = np.array(Image.open(fmap[inp["file_name"]]), dtype=np.int64)
            g = gt.copy(); g[g == ignore] = K
            conf += np.bincount((K + 1) * pred.reshape(-1) + g.reshape(-1),
                                minlength=(K + 1) ** 2).reshape(K + 1, K + 1)
    sub = conf[:K, :K].astype(np.float64)
    tp = np.diag(sub); union = sub.sum(0) + sub.sum(1) - tp
    return np.where(union > 0, tp / np.maximum(union, 1), np.nan)
 
 
def main():
    import train_net
    from detectron2.engine import default_argument_parser
    args = default_argument_parser().parse_args(
        ["--config-file", CONFIG_FILE, "--eval-only", "MODEL.WEIGHTS", WEIGHTS])
    cfg = train_net.setup(args)
 
    freqs = {}
    for d, name in DATASETS.items():
        fmap, classes, ignore = gt_index(name)
        imgf, pixf = class_frequency(fmap, len(classes), ignore)
        freqs[d] = (imgf, pixf, classes, fmap, ignore)
        print(f"  scanned GT for {d}: {len(fmap)} images")
 
    classes = freqs["bdd"][2]
    K = len(classes)
    print(f"\n1) Class frequency  (% images containing class | % of labelled pixels)")
    print(f"   {'class':16s}{'BDD img%':>10s}{'BDD pix%':>10s}{'CS img%':>10s}{'CS pix%':>10s}")
    bi, bp = freqs["bdd"][0], freqs["bdd"][1]
    ci, cp = freqs["cityscapes"][0], freqs["cityscapes"][1]
    order = np.argsort(bi)
    for i in order:
        print(f"   {classes[i]:16s}{bi[i]:10.1f}{bp[i]:10.2f}{ci[i]:10.1f}{cp[i]:10.2f}")
 
    absent = [classes[i] for i in range(K) if bi[i] < ABSENT_IMG_FREQ]
    print(f"\n2) Effectively absent in BDD (<{ABSENT_IMG_FREQ:g}% of images): "
          f"{absent if absent else 'none'}")
 
    # present-class mIoU
    print("\n3) BDD mIoU: standard 19-class vs present-classes-only")
    fmap, _, ignore = freqs["bdd"][3], None, freqs["bdd"][4]
    iou = per_class_iou(cfg, DATASETS["bdd"], fmap, K, ignore)
    present_mask = bi >= ABSENT_IMG_FREQ
    m_all = np.nanmean(iou) * 100
    m_present = np.nanmean(iou[present_mask]) * 100
    print(f"   19-class mIoU        = {m_all:.2f}")
    print(f"   present-class mIoU   = {m_present:.2f}  "
          f"({int(present_mask.sum())} classes >= {ABSENT_IMG_FREQ:g}% img-freq)")
    print(f"   difference from rare-class averaging = {m_present - m_all:+.2f}")
    print("\nREAD: a large positive difference => part of BDD's low mean is rare-class")
    print("averaging (report present-class mIoU too). A small difference => the low mean")
    print("is real per-class difficulty, not a class-composition artifact.")
 
 
if __name__ == "__main__":
    main()

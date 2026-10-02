"""
Recognition-vs-Grouping oracle
==============================
Decomposes a semantic-segmentation model's failure into two independent causes:
 
  GROUPING / localization : are region boundaries placed correctly, regardless of
                            the class name assigned?
  RECOGNITION / naming    : given a correctly-placed region, is the class right?
 
Two oracle upper bounds, per domain:
 
  real  M   : the model's actual mIoU.
  oracleA   : GROUPING ceiling — keep the model's predicted regions, but relabel
              each connected component with the GT class it most overlaps
              (classification made perfect). Loss (100 - oracleA) is pure grouping
              error; gain (oracleA - M) is loss that was due to mislabelling.
  oracleB   : RECOGNITION ceiling — keep perfect GT region boundaries, assign each
              GT connected component the model's majority predicted class
              (grouping made perfect). Loss (100 - oracleB) is pure recognition
              error.
 
Interpretation:
  oracleB high, oracleA ~ M   -> grouping is the bottleneck  (geometry/viewpoint)
  oracleA high, oracleB ~ M   -> recognition is the bottleneck (taxonomy/labels/confusion)
Per-class rows show WHICH classes (e.g. BDD's rider/bus) are grouping- vs
recognition-limited.
 
Uses vanilla (BDD's difficulty is a property of the base model + data).
Run:  python diagnose_recognition_grouping.py
"""
 
import numpy as np
import torch
from PIL import Image
 
try:
    from scipy.ndimage import label as cc_label   # connected components
except Exception:
    cc_label = None
 
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_FILE = "custom_configs/training/vanilla_mask2former_swinb_90k.yaml"
WEIGHTS     = "experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth"
DATASETS    = {"cityscapes": "cityscapes_fine_sem_seg_val",
               "bdd":        "bdd100k_sem_seg_val"}
CAP         = 150     # images per domain (oracle CC is the cost; raise for stability)
 
 
def majority(vals, K):
    vals = vals[vals < K]
    if vals.size == 0:
        return -1
    return int(np.bincount(vals, minlength=K).argmax())
 
 
def oracle_maps(pred, gt, K, ignore):
    """Return (oracleA, oracleB) label maps."""
    oa = pred.copy()
    ob = pred.copy()
    gmask_ignore = (gt == ignore)
 
    # oracleA: predicted connected components relabelled by GT majority
    for c in np.unique(pred):
        comps, n = cc_label(pred == c)
        for i in range(1, n + 1):
            region = comps == i
            gv = gt[region]
            m = majority(gv[gv != ignore] if ignore is not None else gv, K)
            if m >= 0:
                oa[region] = m
 
    # oracleB: GT connected components relabelled by predicted majority
    gt_valid = np.where(gmask_ignore, K, gt)   # ignore -> K (won't form class regions)
    for c in np.unique(gt_valid):
        if c >= K:
            continue
        comps, n = cc_label(gt_valid == c)
        for i in range(1, n + 1):
            region = comps == i
            m = majority(pred[region], K)
            if m >= 0:
                ob[region] = m
    return oa, ob
 
 
def iou_from_conf(conf, K):
    sub = conf[:K, :K].astype(np.float64)
    tp = np.diag(sub); union = sub.sum(0) + sub.sum(1) - tp
    return np.where(union > 0, tp / np.maximum(union, 1), np.nan)
 
 
def accumulate(conf, pred, gt, K, ignore):
    g = gt.copy(); g[g == ignore] = K
    conf += np.bincount((K + 1) * pred.reshape(-1) + g.reshape(-1),
                        minlength=(K + 1) ** 2).reshape(K + 1, K + 1)
 
 
@torch.no_grad()
def run_domain(model, cfg, dataset_name, fmap, K, ignore, names):
    from detectron2.data import build_detection_test_loader
    cM = np.zeros((K + 1, K + 1), np.int64)
    cA = np.zeros((K + 1, K + 1), np.int64)
    cB = np.zeros((K + 1, K + 1), np.int64)
    seen = 0
    for batch in build_detection_test_loader(cfg, dataset_name):
        for inp, out in zip(batch, model(batch)):
            pred = out["sem_seg"].argmax(0).to("cpu").numpy().astype(np.int64)
            gt = np.array(Image.open(fmap[inp["file_name"]]), dtype=np.int64)
            oa, ob = oracle_maps(pred, gt, K, ignore)
            accumulate(cM, pred, gt, K, ignore)
            accumulate(cA, oa, gt, K, ignore)
            accumulate(cB, ob, gt, K, ignore)
            seen += 1
            if seen >= CAP:
                break
        if seen >= CAP:
            break
    return iou_from_conf(cM, K), iou_from_conf(cA, K), iou_from_conf(cB, K), seen
 
 
def main():
    if cc_label is None:
        raise SystemExit("scipy needed: pip install scipy --break-system-packages")
    import train_net
    from detectron2.engine import default_argument_parser
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import DatasetCatalog, MetadataCatalog
 
    args = default_argument_parser().parse_args(
        ["--config-file", CONFIG_FILE, "--eval-only", "MODEL.WEIGHTS", WEIGHTS])
    cfg = train_net.setup(args)
    model = build_model(cfg).eval().to(DEVICE)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
 
    for d, name in DATASETS.items():
        dicts = DatasetCatalog.get(name); meta = MetadataCatalog.get(name)
        fmap = {x["file_name"]: x["sem_seg_file_name"] for x in dicts}
        classes = meta.stuff_classes; K = len(classes)
        ignore = getattr(meta, "ignore_label", 255)
        iM, iA, iB, seen = run_domain(model, cfg, name, fmap, K, ignore, classes)
 
        print(f"\n### {d}  ({seen} imgs)   real vs grouping-ceiling(A) vs recognition-ceiling(B)")
        print(f"  {'class':16s}{'real':>8s}{'oracleA':>9s}{'oracleB':>9s}"
              f"{'grpLoss':>9s}{'recLoss':>9s}")
        for i in np.argsort(np.nan_to_num(iM, nan=1e9)):
            m, a, b = iM[i]*100, iA[i]*100, iB[i]*100
            # grouping loss = 100 - oracleA (label-corrected, so residual is boundaries)
            # recognition loss = 100 - oracleB (region-perfect, so residual is naming)
            print(f"  {classes[i]:16s}{m:8.1f}{a:9.1f}{b:9.1f}{100-a:9.1f}{100-b:9.1f}")
        print(f"  {'mIoU':16s}{np.nanmean(iM)*100:8.1f}{np.nanmean(iA)*100:9.1f}"
              f"{np.nanmean(iB)*100:9.1f}{100-np.nanmean(iA)*100:9.1f}"
              f"{100-np.nanmean(iB)*100:9.1f}")
 
    print("\nREAD: for each domain (esp. BDD) compare the two ceilings.")
    print("  oracleA >> real  => big gains just from correct labels  => RECOGNITION problem")
    print("  oracleB >> real  => big gains just from correct regions => GROUPING problem")
    print("  Per-class: which of BDD's weak classes (rider/bus/bicycle) recover under")
    print("  which oracle tells you whether they're mislocalized or misnamed.")
 
 
if __name__ == "__main__":
    main()

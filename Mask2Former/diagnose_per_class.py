"""
Per-class IoU breakdown  —  vanilla vs CSFD, all four domains
=============================================================
 
Answers, without pasting any logs:
  - Is BDD's low mean a few FLOOR classes (~0 IoU) -> label/taxonomy/composition,
    or a UNIFORM depression across all 19 -> geometry/quality?
  - Where does GTA5's +2.33 actually come from (which classes)?
  - Does CSFD move any individual class, or is it per-class flat (matching the
    inert-module finding)?
 
Reproduces detectron2 SemSegEvaluator exactly: same GT files and ignore label from
your registered datasets, same confusion-matrix accumulation, ignore row/col
excluded via conf[:-1,:-1]. Numbers will match your official mIoU.
 
Run:  python diagnose_per_class.py
GTA5 val is ~25k imgs; it's capped below for speed (per-class IoU is stable by a
few thousand). Set the cap to None for the full set.
"""
 
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
MODELS = {
    "vanilla": ("custom_configs/training/vanilla_mask2former_swinb_90k.yaml",
                "experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth"),
    "csfd":    ("custom_configs/training/logit_adjust_v1_swinb_90k.yaml",
                "experiments/training_outputs/logit_adjust_v1_swinb_90k/model_final.pth"),
}
DOMAIN_DATASETS = {
    "cityscapes": "cityscapes_fine_sem_seg_val",
    "bdd":        "bdd100k_sem_seg_val",
    "gta5":       "gta5_sem_seg_val",
    "mapillary":  "mapillary_vistas_sem_seg_val",
}
DOMAINS = ["cityscapes", "bdd", "gta5", "mapillary"]
PER_DOMAIN_CAP = {"cityscapes": None, "bdd": None, "gta5": 3000, "mapillary": None}
 
 
def build_model(config_file, weights):
    import train_net
    from detectron2.engine import default_argument_parser
    from detectron2.modeling import build_model as d2_build
    from detectron2.checkpoint import DetectionCheckpointer
    args = default_argument_parser().parse_args(
        ["--config-file", config_file, "--eval-only", "MODEL.WEIGHTS", weights])
    cfg = train_net.setup(args)
    model = d2_build(cfg).eval().to(DEVICE)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    return model, cfg
 
 
def gt_index(dataset_name):
    """file_name -> sem_seg_file_name, plus (num_classes, ignore_label, class_names)."""
    from detectron2.data import DatasetCatalog, MetadataCatalog
    dicts = DatasetCatalog.get(dataset_name)
    meta = MetadataCatalog.get(dataset_name)
    fmap = {d["file_name"]: d["sem_seg_file_name"] for d in dicts}
    names = meta.stuff_classes
    ignore = getattr(meta, "ignore_label", 255)
    return fmap, len(names), ignore, names
 
 
@torch.no_grad()
def eval_domain(model, cfg, dataset_name, cap):
    from detectron2.data import build_detection_test_loader
    fmap, K, ignore, names = gt_index(dataset_name)
    conf = np.zeros((K + 1, K + 1), dtype=np.int64)
    loader = build_detection_test_loader(cfg, dataset_name)
    seen = 0
    for batch in loader:
        outputs = model(batch)
        for inp, out in zip(batch, outputs):
            pred = out["sem_seg"].argmax(0).to("cpu")                 # [H, W]
            gt = np.array(Image.open(fmap[inp["file_name"]]), dtype=np.int64)
            if pred.shape != gt.shape:                                 # safety: align to GT
                pred = F.interpolate(pred[None, None].float(), size=gt.shape,
                                     mode="nearest")[0, 0]
            pred = pred.numpy().astype(np.int64)
            g = gt.copy(); g[g == ignore] = K
            conf += np.bincount((K + 1) * pred.reshape(-1) + g.reshape(-1),
                                minlength=(K + 1) ** 2).reshape(K + 1, K + 1)
            seen += 1
            if cap and seen >= cap:
                break
        if cap and seen >= cap:
            break
    return conf, K, names, seen
 
 
def per_class_iou(conf, K):
    sub = conf[:K, :K].astype(np.float64)          # exclude ignore row/col (detectron2)
    tp = np.diag(sub)
    pos_gt = sub.sum(0); pos_pred = sub.sum(1)
    union = pos_gt + pos_pred - tp
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    return iou                                     # [K], nan where class absent
 
 
def main():
    results = {}   # results[model][domain] = (iou[K], names, seen)
    for mname, (cfg_file, weights) in MODELS.items():
        print(f"\n=== building {mname} ===")
        model, cfg = build_model(cfg_file, weights)
        results[mname] = {}
        for d in DOMAINS:
            conf, K, names, seen = eval_domain(model, cfg, DOMAIN_DATASETS[d],
                                               PER_DOMAIN_CAP.get(d))
            results[mname][d] = (per_class_iou(conf, K), names, seen)
            miou = np.nanmean(results[mname][d][0]) * 100
            print(f"  {mname:8s} {d:11s} mIoU={miou:5.2f}  ({seen} imgs)")
        del model
        torch.cuda.empty_cache()
 
    # per-domain per-class table: vanilla | csfd | delta
    for d in DOMAINS:
        iou_v, names, _ = results["vanilla"][d]
        iou_c, _, _ = results["csfd"][d]
        print(f"\n### {d}   (per-class IoU %, sorted by CSFD)")
        print(f"  {'class':16s}{'vanilla':>9s}{'csfd':>9s}{'delta':>9s}")
        order = np.argsort(np.nan_to_num(iou_c, nan=-1))
        for i in order:
            v, c = iou_v[i] * 100, iou_c[i] * 100
            dv = c - v
            print(f"  {names[i]:16s}{v:9.2f}{c:9.2f}{dv:+9.2f}")
        print(f"  {'mIoU':16s}{np.nanmean(iou_v)*100:9.2f}{np.nanmean(iou_c)*100:9.2f}"
              f"{(np.nanmean(iou_c)-np.nanmean(iou_v))*100:+9.2f}")
 
    print("\nREAD: floor classes (~0 in BOTH models) = label/composition problem, not CSFD's.")
    print("Large negative deltas that offset positives = CSFD trading classes (it isn't,")
    print("if the module is inert). Compare BDD's floor classes against GTA5's to see")
    print("whether BDD's low mean is a distinct, source-invariant set of classes.")
 
 
if __name__ == "__main__":
    main()
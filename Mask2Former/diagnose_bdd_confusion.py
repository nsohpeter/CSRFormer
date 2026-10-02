"""
BDD confusion matrix  —  the decision switch
============================================
BDD is recognition-limited (regions found, classes misnamed). This ranks WHICH
class pairs get swapped and, crucially, whether each swap is:
 
  ONE-DIRECTIONAL  (BDD gt=X predicted as Y almost always, but gt=Y rarely -> X)
      => the model is consistently mapping BDD's X onto Cityscapes' Y. Smells like
         an ANNOTATION-PROTOCOL / taxonomy mismatch (BDD defines X differently).
         Fix is label mapping / corrected eval, NOT a new module.
 
  BIDIRECTIONAL / scattered
      => genuine classifier confusion. Fix is a RECOGNITION-targeted objective
         (logit adjustment, class-balanced / confusion-aware loss).
 
Outputs:
  1. row-normalized confusion for BDD present classes (what each true class is
     predicted as, %)
  2. top confused pairs by pixel mass, each with a DIRECTIONALITY ratio
     asym = P(pred=Y | gt=X) / P(pred=X | gt=Y)   (>>1 or <<1 = one-directional)
  3. Cityscapes asymmetry for the same pairs, as a baseline (is the swap BDD-specific?)
 
Uses vanilla. Run:  python diagnose_bdd_confusion.py
"""
 
import numpy as np
import torch
from PIL import Image
 
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_FILE = "custom_configs/training/vanilla_mask2former_swinb_90k.yaml"
WEIGHTS     = "experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth"
DATASETS    = {"bdd": "bdd100k_sem_seg_val", "cityscapes": "cityscapes_fine_sem_seg_val"}
PRESENT_MIN = 5.0    # only analyze classes present in >= this % of BDD images
TOP_PAIRS   = 15
 
 
def gt_index(name):
    from detectron2.data import DatasetCatalog, MetadataCatalog
    dicts = DatasetCatalog.get(name); meta = MetadataCatalog.get(name)
    return ({d["file_name"]: d["sem_seg_file_name"] for d in dicts},
            meta.stuff_classes, getattr(meta, "ignore_label", 255))
 
 
@torch.no_grad()
def confusion(model, cfg, name, fmap, K, ignore):
    from detectron2.data import build_detection_test_loader
    conf = np.zeros((K, K), dtype=np.int64)     # rows = GT, cols = PRED
    img_freq = np.zeros(K, dtype=np.int64); n = 0
    for batch in build_detection_test_loader(cfg, name):
        for inp, out in zip(batch, model(batch)):
            pred = out["sem_seg"].argmax(0).to("cpu").numpy().astype(np.int64)
            gt = np.array(Image.open(fmap[inp["file_name"]]), dtype=np.int64)
            m = gt != ignore
            g, p = gt[m], pred[m]
            conf += np.bincount(K * g + p, minlength=K * K).reshape(K, K)
            img_freq[np.unique(gt[m])] += 1; n += 1
    return conf, img_freq / max(n, 1) * 100.0
 
 
def main():
    import train_net
    from detectron2.engine import default_argument_parser
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    args = default_argument_parser().parse_args(
        ["--config-file", CONFIG_FILE, "--eval-only", "MODEL.WEIGHTS", WEIGHTS])
    cfg = train_net.setup(args)
    model = build_model(cfg).eval().to(DEVICE)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
 
    conf, names, ignore = {}, None, None
    freq = {}
    for d, name in DATASETS.items():
        fmap, cls, ig = gt_index(name); names = cls; ignore = ig
        K = len(cls)
        conf[d], freq[d] = confusion(model, cfg, name, fmap, K, ig)
    K = len(names)
 
    # row-normalized BDD confusion (P(pred | gt))
    Cb = conf["bdd"].astype(np.float64)
    rown = Cb / Cb.sum(1, keepdims=True).clip(1)
    present = [i for i in range(K) if freq["bdd"][i] >= PRESENT_MIN]
 
    print("1) BDD row-normalized confusion  P(pred=col | gt=row), % — present classes")
    print("   (diagonal omitted; showing off-diagonal mass)")
    for i in present:
        conf_to = sorted([(rown[i, j], j) for j in range(K) if j != i], reverse=True)[:3]
        s = "  ".join(f"{names[j]}:{p*100:4.1f}" for p, j in conf_to if p > 0.02)
        print(f"   {names[i]:14s} acc={rown[i,i]*100:4.1f}   -> {s}")
 
    # top confused pairs by off-diagonal pixel mass, with directionality
    print(f"\n2) Top confused pairs (by BDD pixel mass) + directionality")
    print(f"   asym = P(pred=B|gt=A) / P(pred=A|gt=B);  >>1 or <<1 = ONE-DIRECTIONAL")
    print(f"   {'gt':13s}{'->pred':13s}{'massBDD%':>9s}{'asymBDD':>9s}{'asymCS':>9s}  verdict")
    Cs = conf["cityscapes"].astype(np.float64)
    rowCs = Cs / Cs.sum(1, keepdims=True).clip(1)
    tot = Cb.sum()
    pairs = []
    for a in present:
        for b in present:
            if b <= a:
                continue
            mass = (Cb[a, b] + Cb[b, a]) / tot * 100
            pairs.append((mass, a, b))
    for mass, a, b in sorted(pairs, reverse=True)[:TOP_PAIRS]:
        pab, pba = rown[a, b], rown[b, a]
        asym = (pab + 1e-6) / (pba + 1e-6)
        pab_c, pba_c = rowCs[a, b], rowCs[b, a]
        asym_c = (pab_c + 1e-6) / (pba_c + 1e-6)
        oned = asym if asym >= 1 else 1 / asym
        verdict = "ONE-DIR (taxonomy?)" if oned > 3 else "bidirectional (confusion)"
        print(f"   {names[a]:13s}{names[b]:13s}{mass:9.2f}{asym:9.2f}{asym_c:9.2f}  {verdict}")
 
    print("\nREAD: pairs with large mass + high asymmetry that is BDD-specific (asymBDD")
    print("far from 1 while asymCS ~1) => BDD annotates that class differently =>")
    print("taxonomy/label fix. Pairs with asym~1 (both directions) => genuine classifier")
    print("confusion => recognition-loss fix. The mix across top pairs decides the solution.")
 
 
if __name__ == "__main__":
    main()

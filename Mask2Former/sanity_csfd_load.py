"""
CSFD load sanity check — v2 run
===============================
Confirms the newly trained csfd_v2 checkpoint actually loads to a real, trained
model before you delete snapshots or trust any diagnostic.
 
Prints:
  1. checkpoint missing / unexpected keys (should be none)
  2. gate last-conv |w| + bias, robustifier last-conv |w|
     (TRAINED model => gate |w| and robustifier |w| clearly NONZERO;
      all-zero + bias 1.0 = untrained/at-init, i.e. the dead-file symptom)
  3. Cityscapes mIoU  (expect ~78-80; ~1 = broken/untrained)
 
Run:  python sanity_csfd_load_v2.py
"""
 
import io, logging
import numpy as np
import torch
from PIL import Image
 
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_FILE = "custom_configs/training/csfd_v2_swinb_90k.yaml"
WEIGHTS     = "experiments/training_outputs/csfd_v2_swinb_90k/model_final.pth"
CS_DATASET  = "cityscapes_fine_sem_seg_val"
 
 
def main():
    import train_net
    from detectron2.engine import default_argument_parser
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
 
    args = default_argument_parser().parse_args(
        ["--config-file", CONFIG_FILE, "--eval-only", "MODEL.WEIGHTS", WEIGHTS])
    cfg = train_net.setup(args)
    model = build_model(cfg).eval().to(DEVICE)
 
    buf = io.StringIO(); h = logging.StreamHandler(buf)
    for lname in ("fvcore.common.checkpoint", "detectron2.checkpoint",
                  "d2.checkpoint.detection_checkpoint"):
        logging.getLogger(lname).addHandler(h); logging.getLogger(lname).setLevel(logging.INFO)
    DetectionCheckpointer(model).load(WEIGHTS)
 
    print("\n=== 1) checkpoint load messages ===")
    msg = buf.getvalue().strip()
    print(msg if msg else "(no incompatible-key messages)")
 
    csfd = next((m for _, m in model.named_modules()
                 if m.__class__.__name__ == "CSFD"), None)
    print("\n=== 2) CSFD parameter inspection ===")
    if csfd is None:
        print("  NO CSFD module in the built model.")
    else:
        gate = next(m for _, m in csfd.named_modules() if m.__class__.__name__ == "SpatialGate")
        gconv = [m for m in gate.gate_net if isinstance(m, torch.nn.Conv2d)][-1]
        rob = next(m for _, m in csfd.named_modules() if m.__class__.__name__ == "StyleRobustifier")
        rconv = [m for m in rob.transform if isinstance(m, torch.nn.Conv2d)][-1]
        print(f"  gate last-conv  |w|={gconv.weight.norm().item():.4e}  bias={gconv.bias.mean().item():.4f}")
        print(f"  robustifier last-conv |w|={rconv.weight.norm().item():.4e}")
        print("  => TRAINED if both |w| clearly nonzero; all-zero+bias1.0 = untrained.")
 
    dicts = DatasetCatalog.get(CS_DATASET); meta = MetadataCatalog.get(CS_DATASET)
    fmap = {d["file_name"]: d["sem_seg_file_name"] for d in dicts}
    K = len(meta.stuff_classes); ignore = getattr(meta, "ignore_label", 255)
    conf = np.zeros((K + 1, K + 1), dtype=np.int64)
    loader = build_detection_test_loader(cfg, CS_DATASET)
    with torch.no_grad():
        for batch in loader:
            for inp, out in zip(batch, model(batch)):
                pred = out["sem_seg"].argmax(0).to("cpu").numpy().astype(np.int64)
                gt = np.array(Image.open(fmap[inp["file_name"]]), dtype=np.int64)
                g = gt.copy(); g[g == ignore] = K
                conf += np.bincount((K + 1) * pred.reshape(-1) + g.reshape(-1),
                                    minlength=(K + 1) ** 2).reshape(K + 1, K + 1)
    sub = conf[:K, :K].astype(np.float64)
    tp = np.diag(sub); union = sub.sum(0) + sub.sum(1) - tp
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    print("\n=== 3) Cityscapes mIoU ===")
    print(f"  csfd_v2 Cityscapes mIoU = {np.nanmean(iou)*100:.2f}   (expect ~78-80)")
 
 
if __name__ == "__main__":
    main()
 

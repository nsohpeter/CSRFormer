"""
Checkpoint key-match diagnostic
===============================
The CSFD model loads to 1% on Cityscapes with its gate/robustifier at exact init,
while its backbone-derived features look fine. That means part of the checkpoint
(the head / csfd subtree) is silently not matching the built model's keys.
 
This lists, per subtree, how many of the model's parameters are actually present
(and shape-matched) in the checkpoint vs. missing — so we see exactly what fails
to load, and why the head is random.
 
Run:  python diagnose_ckpt_keys.py
"""
 
import collections
import torch
 
CONFIG_FILE = "custom_configs/training/csfd_swinb_90k.yaml"
WEIGHTS     = "experiments/training_outputs/csfd_swinb_90k/model_final.pth"
 
 
def subtree(key):
    if key.startswith("backbone"):                      return "backbone"
    if key.startswith("sem_seg_head.pixel_decoder"):    return "sem_seg_head.pixel_decoder"
    if key.startswith("sem_seg_head.predictor"):        return "sem_seg_head.predictor"
    if key.startswith("sem_seg_head.csfd"):             return "sem_seg_head.csfd"
    if key.startswith("sem_seg_head"):                  return "sem_seg_head.(other)"
    return key.split(".")[0]
 
 
def main():
    import train_net
    from detectron2.engine import default_argument_parser
    from detectron2.modeling import build_model
 
    args = default_argument_parser().parse_args(
        ["--config-file", CONFIG_FILE, "--eval-only", "MODEL.WEIGHTS", WEIGHTS])
    cfg = train_net.setup(args)
    model = build_model(cfg)
    msd = model.state_dict()
 
    raw = torch.load(WEIGHTS, map_location="cpu")
    print("checkpoint top-level keys:", list(raw.keys())[:8])
    csd = raw.get("model", raw)
    # strip common prefixes if present
    sample = next(iter(csd))
    print("sample checkpoint param key:", sample)
    print("sample model param key:     ", next(iter(msd)))
    print(f"checkpoint has {len(csd)} params; model has {len(msd)} params\n")
 
    ck_keys = set(csd.keys())
    # try to detect a uniform prefix mismatch (e.g. 'module.' or missing)
    stripped = {k.split("module.", 1)[-1] for k in ck_keys}
 
    stats = collections.defaultdict(lambda: [0, 0, 0])  # subtree -> [total, matched, shape_mismatch]
    missing_examples = collections.defaultdict(list)
    for k, v in msd.items():
        st = subtree(k)
        stats[st][0] += 1
        src = None
        if k in csd:               src = csd[k]
        elif ("module." + k) in csd: src = csd["module." + k]
        if src is None:
            if len(missing_examples[st]) < 3:
                missing_examples[st].append(k)
            continue
        if tuple(src.shape) == tuple(v.shape):
            stats[st][1] += 1
        else:
            stats[st][2] += 1
            if len(missing_examples[st]) < 3:
                missing_examples[st].append(f"{k}  ckpt{tuple(src.shape)} vs model{tuple(v.shape)}")
 
    print(f"{'subtree':32s}{'model#':>8s}{'matched':>9s}{'shapemis':>9s}{'missing':>9s}")
    for st in sorted(stats):
        tot, matched, mis = stats[st]
        print(f"{st:32s}{tot:8d}{matched:9d}{mis:9d}{tot-matched-mis:9d}")
        for ex in missing_examples[st]:
            print(f"    e.g. {ex}")
 
    # also: checkpoint keys not used by the model
    model_keys_all = set(msd) | {"module." + k for k in msd}
    unused = [k for k in ck_keys if k not in model_keys_all]
    print(f"\ncheckpoint keys NOT matched to any model param: {len(unused)}")
    for k in unused[:10]:
        print("   unused:", k)
 
 
if __name__ == "__main__":
    main()


"""
Checkpoint integrity audit
==========================
A botched resume overwrote csfd_swinb_90k/model_final.pth with an iteration-4
(untrained) snapshot. This scans every model_final.pth under experiments/ and
reports its training iteration, so you can see at a glance which result-backing
checkpoints are real (iteration ~ the configured schedule, e.g. 89999/90000) and
which were clobbered (tiny iteration).
 
Reads headers on CPU; no GPU, no model build. A few seconds per 400MB file.
 
Run:  python audit_checkpoints.py
"""
 
import glob, os, torch
 
ROOTS = ["experiments/training_outputs", "experiments/gta5_training_outputs",
         "experiments"]
SUSPECT_BELOW = 1000   # iterations below this almost certainly = untrained/clobbered
 
 
def norm_of(sd, needle):
    for k, v in sd.items():
        if needle in k and k.endswith(".weight") and hasattr(v, "dim") and v.dim() > 1:
            return float(v.norm())
    return float("nan")
 
 
def main():
    seen = set()
    paths = []
    for r in ROOTS:
        paths += glob.glob(os.path.join(r, "**", "model_final.pth"), recursive=True)
    paths = sorted(set(paths))
 
    print(f"{'iteration':>10s}  {'predictor|w|':>12s}  {'csfd.gate|w|':>12s}  path")
    for p in paths:
        rp = os.path.realpath(p)
        if rp in seen:
            continue
        seen.add(rp)
        try:
            c = torch.load(p, map_location="cpu")
        except Exception as e:
            print(f"{'ERR':>10s}  {'':>12s}  {'':>12s}  {p}   ({e})")
            continue
        it = c.get("iteration", None)
        sd = c.get("model", c)
        pn = norm_of(sd, "predictor.transformer_self_attention_layers.0.self_attn.out_proj")
        gn = norm_of(sd, "csfd.gate")
        flag = "  <-- SUSPECT (untrained?)" if (it is not None and it < SUSPECT_BELOW) else ""
        gs = f"{gn:12.3f}" if gn == gn else f"{'--':>12s}"
        print(f"{str(it):>10s}  {pn:12.3f}  {gs}  {p}{flag}")
 
    print("\nA trained 90k run should show iteration ~89999/90000. Anything with a tiny")
    print("iteration was saved mid-restart and is NOT the model your results came from.")
    print("For CSFD runs, csfd.gate|w| = 0.000 means the gate is at init (untrained OR")
    print("genuinely-inert — only distinguishable once iteration confirms it's trained).")
 
 
if __name__ == "__main__":
    main()

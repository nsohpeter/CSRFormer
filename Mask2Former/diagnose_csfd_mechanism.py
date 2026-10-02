"""
CSFD Mechanism Diagnostic  —  "what is CSFD doing, and why does it help GTA5/Map
but not BDD?"
================================================================================
 
Hooks the trained CSFD's INPUT, OUTPUT, and GATE, and for each domain measures:
 
  1. gate_mean, |dfeat|  : is the module even ACTIVE on this domain, or inert?
       gate_mean ~ pass-through init (~0.73) and |dfeat|~0  => module does nothing here.
  2. meanGap / stdGap (source vs target), on INPUT vs OUTPUT features:
       does CSFD pull the target's feature statistics TOWARD Cityscapes?
       ratio = gap_output / gap_input.  <1 = CSFD reduces the domain gap; ~1 = no effect.
 
Central hypothesis being tested:
    CSFD reduces the CS<->GTA5 and CS<->Mapillary gaps (appearance-dominated, which
    its IN-on-high-freq addresses) but leaves the CS<->BDD gap ~unchanged (BDD's
    binding constraint is not appearance) -> which is exactly why it doesn't help BDD.
 
The gap metric is a CONTINUOUS mean-shift distance in a standardized descriptor
space (linear-MMD style). Unlike a classifier-AUC it does NOT saturate at 1.0, so
it can actually show a gap SHRINKING.  meanGap = first-order (mean) shift;
stdGap = second-order (per-channel std) shift — CSFD's IN targets the latter.
 
Wiring is identical to diagnose_frequency_bands.py (same paths, same loaders).
Run:  python diagnose_csfd_mechanism.py
"""
 
import math
import torch
 
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_PER_D = 300
BATCHES_CAP  = None
SOURCE       = "cityscapes"
 
CONFIG_FILE  = "custom_configs/training/csfd_v2_swinb_90k.yaml"
WEIGHTS      = "experiments/training_outputs/csfd_v2_swinb_90k/model_final.pth"
DOMAIN_DATASETS = {
    "cityscapes": "cityscapes_fine_sem_seg_val",
    "bdd":        "bdd100k_sem_seg_val",
    "gta5":       "gta5_sem_seg_val",
    "mapillary":  "mapillary_vistas_sem_seg_val",
}
DOMAINS = ["cityscapes", "bdd", "gta5", "mapillary"]
 
 
def build_model_and_hooks():
    """Build trained model; hook CSFD input (pre), output (fwd), and the gate map."""
    import train_net
    from detectron2.engine import default_argument_parser
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
 
    args = default_argument_parser().parse_args(
        ["--config-file", CONFIG_FILE, "--eval-only", "MODEL.WEIGHTS", WEIGHTS])
    cfg = train_net.setup(args)
    model = build_model(cfg).eval().to(DEVICE)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
 
    csfd = next((m for _, m in model.named_modules()
                 if m.__class__.__name__ == "CSFD"), None)
    if csfd is None:
        raise RuntimeError("No CSFD module found — check the config enables it.")
    gate = next((m for _, m in csfd.named_modules()
                 if m.__class__.__name__ == "SpatialGate"), None)
    robust = next((m for _, m in csfd.named_modules()
                   if m.__class__.__name__ == "StyleRobustifier"), None)
 
    h = {}
    csfd.register_forward_pre_hook(lambda m, a: h.__setitem__("in", a[0].detach()))
    csfd.register_forward_hook(lambda m, a, o: h.__setitem__("out", o.detach()))
    if gate is not None:
        gate.register_forward_hook(lambda m, a, o: h.__setitem__("gate", o.detach()))
    if robust is not None:
        # capture both the robustifier's input (style band) and output
        robust.register_forward_pre_hook(lambda m, a: h.__setitem__("rob_in", a[0].detach()))
        robust.register_forward_hook(lambda m, a, o: h.__setitem__("rob_out", o.detach()))
    print(f"[hook] CSFD hooked (gate={gate is not None}, robustifier={robust is not None})")
    return model, h, cfg
 
 
def domain_loaders(cfg):
    from detectron2.data import build_detection_test_loader
    return {d: build_detection_test_loader(cfg, DOMAIN_DATASETS[d]) for d in DOMAINS}
 
 
def descriptor(feat):
    """Per-image [mean_c, std_c] -> [B, 2C]  (BN-style style statistics)."""
    return torch.cat([feat.mean(dim=[2, 3]), feat.std(dim=[2, 3])], dim=1)
 
 
@torch.no_grad()
def collect(model, h, loaders):
    C = None
    store = {d: {"in": [], "out": [], "gate": [], "gate_std": [],
                 "dfeat": [], "rob": []} for d in DOMAINS}
    for d in DOMAINS:
        seen = 0
        for bi, batch in enumerate(loaders[d]):
            if BATCHES_CAP and bi >= BATCHES_CAP:
                break
            model(batch)
            xin, xout = h["in"].float(), h["out"].float()
            store[d]["in"].append(descriptor(xin).cpu())
            store[d]["out"].append(descriptor(xout).cpu())
            dr = (xout - xin).flatten(1).norm(dim=1) / xin.flatten(1).norm(dim=1).clamp_min(1e-6)
            store[d]["dfeat"].append(dr.cpu())
            if "gate" in h:
                g = h["gate"].float()
                store[d]["gate"].append(g.mean(dim=[1, 2, 3]).cpu())
                store[d]["gate_std"].append(g.std(dim=[1, 2, 3]).cpu())   # spatial variation
            if "rob_in" in h:
                ri, ro = h["rob_in"].float(), h["rob_out"].float()
                rr = (ro - ri).flatten(1).norm(dim=1) / ri.flatten(1).norm(dim=1).clamp_min(1e-6)
                store[d]["rob"].append(rr.cpu())                          # IN/transform contribution
            seen += xin.shape[0]
            if seen >= IMAGES_PER_D:
                break
        for k in ("in", "out", "dfeat", "gate", "gate_std", "rob"):
            store[d][k] = torch.cat(store[d][k], 0)[:IMAGES_PER_D] if store[d][k] else None
        print(f"  collected {seen:4d} imgs for {d}")
    return store
 
 
def split_gap(S, T):
    """Continuous mean-shift distance between descriptor sets, standardized.
    Returns (mean_gap, std_gap): first-order and second-order shift in std units.
    Non-saturating (unlike AUC), so gap reductions are visible."""
    S, T = S.double(), T.double()
    C = S.shape[1] // 2
    allX = torch.cat([S, T], 0)
    mu, sd = allX.mean(0), allX.std(0) + 1e-6
    S, T = (S - mu) / sd, (T - mu) / sd
    d = (S.mean(0) - T.mean(0))                # [2C]
    mean_gap = d[:C].norm().item()             # mean-channel block
    std_gap  = d[C:].norm().item()             # std-channel block
    return mean_gap, std_gap
 
 
def main():
    model, h, cfg = build_model_and_hooks()
    loaders = domain_loaders(cfg)
    print("Collecting CSFD input/output/gate per domain...")
    store = collect(model, h, loaders)
    targets = [d for d in DOMAINS if d != SOURCE]
 
    # --- activity: is the module doing anything, per domain? ---
    print("\n1) Module activity per domain")
    print("   {:12s}{:>10s}{:>10s}{:>10s}{:>10s}".format(
          "", "gate_mean", "gate_std", "|dfeat|", "robustif"))
    for d in DOMAINS:
        g  = store[d]["gate"].mean().item()     if store[d]["gate"]     is not None else float("nan")
        gs = store[d]["gate_std"].mean().item() if store[d]["gate_std"] is not None else float("nan")
        df = store[d]["dfeat"].mean().item()
        rb = store[d]["rob"].mean().item()      if store[d]["rob"]      is not None else float("nan")
        print("   {:12s}{:10.3f}{:10.3f}{:10.3f}{:10.3f}".format(d, g, gs, df, rb))
    print("   gate_mean~0.731 = init value (sigmoid(1.0)); gate_std~0 = spatially CONSTANT")
    print("   robustif = ||robustifier_out - style|| / ||style|| ; ~0 => IN/transform branch inert")
    print("   => if gate is constant AND robustif~0, the module = a FIXED high-freq")
    print("      attenuation (out = content + 0.731*style), no learned/adaptive behavior.")
 
    # --- alignment: does CSFD pull target stats toward source? ---
    print("\n2) Source<->target feature gap  BEFORE (input) vs AFTER (output) CSFD")
    print("   ratio<1 => CSFD shrinks the gap (aligns to source);  ~1 => no effect")
    print("   {:12s} | {:>22s} | {:>22s}".format("target",
          "meanGap in->out (ratio)", "stdGap in->out (ratio)"))
    for t in targets:
        mi, si = split_gap(store[SOURCE]["in"],  store[t]["in"])
        mo, so = split_gap(store[SOURCE]["out"], store[t]["out"])
        mr = mo / mi if mi > 1e-9 else float("nan")
        sr = so / si if si > 1e-9 else float("nan")
        print("   {:12s} | {:7.3f} -> {:6.3f} ({:.2f}) | {:7.3f} -> {:6.3f} ({:.2f})"
              .format(t, mi, mo, mr, si, so, sr))
 
    print("\nREAD:")
    print("  If GTA5/Mapillary ratios are <1 (esp. stdGap) but BDD's are ~1, then CSFD")
    print("  aligns the appearance-dominated targets to source and leaves BDD's gap")
    print("  intact -> BDD's constraint is not what CSFD removes. Cross-check with the")
    print("  per-class breakdown, attribute split, and recognition/grouping oracle.")
 
 
if __name__ == "__main__":
    main()
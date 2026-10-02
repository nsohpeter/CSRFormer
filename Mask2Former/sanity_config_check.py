"""
Pre-training config + CSFD sanity check
=======================================
Builds the model from the v2 config (no checkpoint, no training) and verifies:
  1. key config fields are what you intend (dataset, iters, LR, batch, CSFD cfg,
     checkpoint period, output dir, backbone weights)
  2. the CSFD module is actually inserted, at the expected place, with the
     expected param count, and at correct near-identity INIT
  3. a forward pass runs and CSFD preserves channel magnitudes (einsum-safe)
 
Run:  python sanity_config_check.py
"""
 
import torch
 
CONFIG_FILE = "custom_configs/training/csfd_v2_swinb_90k.yaml"
 
 
def main():
    import train_net
    from detectron2.engine import default_argument_parser
    from detectron2.modeling import build_model
 
    # build cfg exactly as training will (no MODEL.WEIGHTS override needed here)
    args = default_argument_parser().parse_args(["--config-file", CONFIG_FILE])
    cfg = train_net.setup(args)
 
    print("=== 1) config fields ===")
    print(f"  OUTPUT_DIR            {cfg.OUTPUT_DIR}")
    print(f"  DATASETS.TRAIN        {cfg.DATASETS.TRAIN}")
    print(f"  DATASETS.TEST         {cfg.DATASETS.TEST}")
    print(f"  SOLVER.MAX_ITER       {cfg.SOLVER.MAX_ITER}")
    print(f"  SOLVER.IMS_PER_BATCH  {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  SOLVER.BASE_LR        {cfg.SOLVER.BASE_LR}")
    print(f"  SOLVER.CHECKPOINT_PERIOD {cfg.SOLVER.CHECKPOINT_PERIOD}")
    print(f"  MODEL.WEIGHTS         {cfg.MODEL.WEIGHTS}")
    try:
        print(f"  MODEL.CSFD.ENABLED    {cfg.MODEL.CSFD.ENABLED}")
        print(f"  MODEL.CSFD.KERNEL_SIZE {cfg.MODEL.CSFD.KERNEL_SIZE}  "
              f"INIT_SIGMA {cfg.MODEL.CSFD.INIT_SIGMA}  REDUCTION {cfg.MODEL.CSFD.REDUCTION}")
    except Exception as e:
        print(f"  MODEL.CSFD  <-- NOT IN CONFIG NAMESPACE ({e})")
 
    # sanity flags on the config itself
    warn = []
    if cfg.SOLVER.CHECKPOINT_PERIOD >= cfg.SOLVER.MAX_ITER:
        warn.append("CHECKPOINT_PERIOD >= MAX_ITER: no intermediate snapshots will be saved")
    if "csfd_v2" not in cfg.OUTPUT_DIR:
        warn.append(f"OUTPUT_DIR is '{cfg.OUTPUT_DIR}' — not the v2 dir you intended")
    if not str(cfg.MODEL.WEIGHTS).endswith(".pkl"):
        warn.append(f"MODEL.WEIGHTS is '{cfg.MODEL.WEIGHTS}' — expected the ImageNet .pkl for from-scratch")
 
    print("\n=== 2) CSFD module in the built model ===")
    model = build_model(cfg)
    csfd_hits = [(n, m) for n, m in model.named_modules() if m.__class__.__name__ == "CSFD"]
    if not csfd_hits:
        print("  !!! NO CSFD module in the built model — config did not insert it.")
    else:
        name, csfd = csfd_hits[0]
        n_params = sum(p.numel() for p in csfd.parameters())
        print(f"  found at: {name}")
        print(f"  params:   {n_params:,}  (expected 179,457 for kernel_size=7, C=256)")
        gate = next(m for _, m in csfd.named_modules() if m.__class__.__name__ == "SpatialGate")
        gconv = [m for m in gate.gate_net if isinstance(m, torch.nn.Conv2d)][-1]
        rob = next(m for _, m in csfd.named_modules() if m.__class__.__name__ == "StyleRobustifier")
        rconv = [m for m in rob.transform if isinstance(m, torch.nn.Conv2d)][-1]
        print(f"  gate last-conv |w|={gconv.weight.norm():.3e} bias={gconv.bias.mean():.3f}"
              f"  (init expected: |w|=0, bias=1.0)")
        print(f"  robustifier last-conv |w|={rconv.weight.norm():.3e}  (init expected: 0)")
 
        # 3) forward pass on the CSFD module: shape + channel-magnitude preservation
        print("\n=== 3) CSFD forward (einsum-safety) ===")
        csfd_cpu = csfd.to("cpu").eval()
        x = torch.randn(1, csfd.in_channels if hasattr(csfd, "in_channels") else 256, 64, 128)
        with torch.no_grad():
            y = csfd_cpu(x)
        in_mag = x.mean(dim=[0, 2, 3]); out_mag = y.mean(dim=[0, 2, 3])
        corr = torch.corrcoef(torch.stack([in_mag, out_mag]))[0, 1]
        print(f"  {tuple(x.shape)} -> {tuple(y.shape)}   channel-mag corr={corr:.5f} "
              f"(want ~1.0)  init |dfeat|={(y-x).abs().mean():.4f}")
 
    if warn:
        print("\n=== WARNINGS ===")
        for w in warn:
            print("  ! " + w)
    else:
        print("\nAll config checks passed.")
 
 
if __name__ == "__main__":
    main()
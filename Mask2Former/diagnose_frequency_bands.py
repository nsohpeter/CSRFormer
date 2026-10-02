"""
Frequency-Band Domain-Style Diagnostic  (premise test for Idea 4)
=================================================================
 
Question this answers
---------------------
Idea 4 assumes the OPTIMAL content/style cutoff is DOMAIN-DEPENDENT: that
different target domains carry their style energy at different spatial
frequencies (e.g. GTA5 rendering artifacts high-freq, ACDC fog low-freq,
BDD camera texture somewhere between). If that's true, one fixed cutoff is
wrong and a multi-band / adaptive decomposition can help. If every target's
style sits in the SAME band, adaptivity buys nothing and you should NOT
spend training runs on it.
 
What it does  (inference only — NO retraining, NO labels)
---------------------------------------------------------
1. Grab the feature map fed INTO CSFD (the mask_features) for a few hundred
   images from the source (Cityscapes) and each target domain.
2. Decompose each map into disjoint Laplacian bands using a FIXED analytical
   Gaussian pyramid (we probe the raw signal, not the trained filter).
3. Per band, summarize each image by per-channel [mean, std] (the classic
   style descriptor).
4. Measure how separable source-vs-target is FROM THAT BAND ALONE, via a
   dependency-free LDA/AUC probe. High separability => that band carries the
   domain shift.
 
Read the output
---------------
A table AUC[band, target]. For each target, the band with the highest AUC is
where its style concentrates.
  - Peak band DIFFERS across targets  -> premise holds, build multi-band.
  - Peak band SAME for all targets     -> one cutoff suffices, skip Idea 4.
Also prints, per target, whether the peak band matches Cityscapes' own
self-band energy (a sanity check that we're seeing shift, not just texture).
 
Wiring (only two things to edit — marked  >>> EDIT)
---------------------------------------------------
  build_model_and_hook()  : build your trained Mask2Former+CSFD, load the
                            checkpoint, return (model, feature_holder). The
                            hook must capture the tensor ENTERING CSFD.
  domain_loaders()        : return {domain_name: iterable_of_image_batches}.
                            Reuse your existing eval dataloaders. Images only;
                            labels are not needed.
 
Run:  python diagnose_frequency_bands.py
"""
 
import math
from collections import defaultdict
 
import torch
import torch.nn.functional as F
 
# ----------------------------------------------------------------------------
# Config  (paths filled in for Peter's repo — adjust if you moved things)
# ----------------------------------------------------------------------------
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_PER_D = 400          # >= 400 recommended for C=256 (held-out probe stability)
BATCHES_CAP  = None         # or an int to cap batches per domain for a quick pass
BAND_SIGMAS  = [1.0, 2.0, 4.0]   # per-stage sigmas, applied successively (Gaussian pyramid)
SOURCE       = "cityscapes"
 
# Your trained v1 CSFD model:
CONFIG_FILE  = "custom_configs/training/csfd_swinb_90k.yaml"
WEIGHTS      = "experiments/training_outputs/csfd_swinb_90k/model_final.pth"
 
# Registered dataset names (confirmed from your experiments/.../miou_scores/*.log):
DOMAIN_DATASETS = {
    "cityscapes": "cityscapes_fine_sem_seg_val",
    "bdd":        "bdd100k_sem_seg_val",
    "gta5":       "gta5_sem_seg_val",
    "mapillary":  "mapillary_vistas_sem_seg_val",
}
DOMAINS = ["cityscapes", "bdd", "gta5", "mapillary"]   # source first; add "acdc" if you want
 
 
# ----------------------------------------------------------------------------
# >>> Step 2a: build your trained model and hook the CSFD input
# ----------------------------------------------------------------------------
def build_model_and_hook():
    """Build the trained Mask2Former+CSFD model and attach a hook that captures
    the tensor entering CSFD (the mask_features) on every forward.
 
    Reuses your own train_net.setup() so all custom config keys (MODEL.CSFD.*)
    and dataset registrations are applied exactly as in training/eval."""
    import train_net                                   # your repo's train_net.py
    from detectron2.engine import default_argument_parser
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
 
    args = default_argument_parser().parse_args([
        "--config-file", CONFIG_FILE,
        "--eval-only",
        "MODEL.WEIGHTS", WEIGHTS,
    ])
    cfg = train_net.setup(args)                         # builds + freezes cfg, registers datasets
 
    model = build_model(cfg)
    model.eval().to(DEVICE)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
 
    # Locate the CSFD instance by class name (no need to know its attribute path).
    csfd_matches = [(n, m) for n, m in model.named_modules()
                    if m.__class__.__name__ == "CSFD"]
    if not csfd_matches:
        raise RuntimeError("No CSFD module found in the built model. Check that "
                           f"{CONFIG_FILE} actually enables CSFD.")
    name, csfd = csfd_matches[0]
    print(f"[hook] CSFD found at: {name}")
 
    holder = {}
    csfd.register_forward_pre_hook(
        lambda module, args: holder.__setitem__("feat", args[0].detach())
    )
    return model, holder, cfg
 
 
# ----------------------------------------------------------------------------
# >>> Step 2b: build one eval loader per domain (labels unused)
# ----------------------------------------------------------------------------
def domain_loaders(cfg):
    """Return {domain: detectron2 test loader}. Reuses your registered datasets."""
    from detectron2.data import build_detection_test_loader
    return {d: build_detection_test_loader(cfg, DOMAIN_DATASETS[d]) for d in DOMAINS}
 
 
def _unused_old_build_model_and_hook():
    """
    Original sketch, kept for reference:
 
        from detectron2.checkpoint import DetectionCheckpointer
        from mask2former import add_maskformer2_config   # your builder
        cfg = setup(...)                                  # your cfg
        model = build_model(cfg).eval().to(DEVICE)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
 
        # Find the CSFD module instance in the pixel-decoder->transformer path:
        csfd = model.sem_seg_head.predictor.csfd   # <-- your actual attribute
 
        holder = {}
        def pre_hook(module, args):
            holder['feat'] = args[0].detach()      # input to CSFD.forward
        csfd.register_forward_pre_hook(pre_hook)
        return model, holder
    """
    pass
 
 
# ----------------------------------------------------------------------------
# Fixed analytical Gaussian pyramid (probes the raw signal)
# ----------------------------------------------------------------------------
def gaussian_kernel(sigma, device=DEVICE):
    """Normalized 2D Gaussian, auto-sized to the sigma (half-width = 3*sigma).
    Auto-sizing matters: a fixed small kernel truncates large-sigma blurs, so the
    coarse bands would collapse into each other and the diagnostic would be blind
    to low-frequency (e.g. fog) style."""
    k = 2 * int(math.ceil(3 * sigma)) + 1
    c = k // 2
    x = torch.arange(k, device=device, dtype=torch.float32) - c
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = (g[:, None] * g[None, :])
    return (g / g.sum())
 
def blur(x, kern):
    C = x.shape[1]
    w = kern[None, None].expand(C, 1, *kern.shape)
    return F.conv2d(x, w, padding=kern.shape[-1] // 2, groups=C)
 
def laplacian_pyramid(x, sigmas):
    """Return (bands, content). bands[k] = blur_{k-1} - blur_k (finest first);
    content = deepest blur (lowest frequency)."""
    prev, bands = x, []
    for s in sigmas:
        cur = blur(prev, gaussian_kernel(s))
        bands.append(prev - cur)
        prev = cur
    return bands, prev
 
def laplacian_bands(x, sigmas):
    return laplacian_pyramid(x, sigmas)[0]
 
 
def style_spectrum(feat, sigmas):
    """Scale-INVARIANT frequency signature of a feature map.
 
    Returns per image:
      prof [B, K]  : fraction of STYLE-band power in each band (sums to 1 over K).
                     Global magnitude cancels, so this isolates WHERE (which
                     frequency) the energy sits — the actual Idea-4 question.
      style_frac [B]: style-band power / (style + content power). How much of the
                     signal is high-freq "style" vs low-freq "content" overall.
 
    Band power uses mean(band^2) (bands are ~zero-mean residuals => power=variance).
    Content uses var() to ignore the DC offset.
    """
    bands, content = laplacian_pyramid(feat, sigmas)
    e = torch.stack([b.pow(2).mean(dim=[1, 2, 3]) for b in bands], dim=1)   # [B, K]
    prof = e / e.sum(dim=1, keepdim=True).clamp_min(1e-12)
    style_e = e.sum(dim=1)
    content_e = content.var(dim=[1, 2, 3])
    style_frac = style_e / (style_e + content_e).clamp_min(1e-12)
    return prof, style_frac
 
 
def style_descriptor(band, use_mean=False):
    """(legacy) per-channel std descriptor. Kept for reference; the raw-separability
    probe built on this SATURATES when the domain shift is large and broadband
    (every band reads AUC 1.0), which is why the primary metric is style_spectrum."""
    sd = band.std(dim=[2, 3])
    if use_mean:
        return torch.cat([band.mean(dim=[2, 3]), sd], dim=1)
    return sd
 
 
# ----------------------------------------------------------------------------
# Dependency-free separability: LDA direction + AUC
# ----------------------------------------------------------------------------
def _auc(ps, pt):
    """Direction-agnostic AUC via rank-sum (Mann-Whitney)."""
    scores = torch.cat([ps, pt])
    ranks = scores.argsort().argsort().double() + 1
    n_s, n_t = ps.numel(), pt.numel()
    auc = (ranks[n_s:].sum() - n_t * (n_t + 1) / 2) / (n_s * n_t)
    return float(max(auc, 1 - auc))
 
def lda_auc(feat_s, feat_t, ridge=1e-1, folds=5):
    """Cross-validated AUC of the best LINEAR separator of source-vs-target
    style descriptors. The LDA direction is fit on a TRAIN split and scored on a
    HELD-OUT split — without this, a high-dim descriptor with few images
    separates sampling noise and inflates every band toward 1.0.
 
    ~0.5 => that band does NOT distinguish the domains (no shift there).
    ->1.0 => that band strongly carries the domain shift.
    """
    Xs, Xt = feat_s.double(), feat_t.double()
    allX = torch.cat([Xs, Xt], 0)
    mu, sd = allX.mean(0), allX.std(0) + 1e-6
    Xs, Xt = (Xs - mu) / sd, (Xt - mu) / sd
    Dd = Xs.shape[1]
    ns, nt = Xs.shape[0], Xt.shape[0]
    perm_s = torch.randperm(ns); perm_t = torch.randperm(nt)
    aucs = []
    for f in range(folds):
        is_te_s = torch.zeros(ns, dtype=torch.bool); is_te_s[perm_s[f::folds]] = True
        is_te_t = torch.zeros(nt, dtype=torch.bool); is_te_t[perm_t[f::folds]] = True
        tr_s, te_s = (~is_te_s).nonzero(as_tuple=True)[0], is_te_s.nonzero(as_tuple=True)[0]
        tr_t, te_t = (~is_te_t).nonzero(as_tuple=True)[0], is_te_t.nonzero(as_tuple=True)[0]
        if len(te_s) == 0 or len(te_t) == 0 or len(tr_s) < 2 or len(tr_t) < 2:
            continue
        A, Bx = Xs[tr_s], Xt[tr_t]
        cov = (torch.cov(A.T) + torch.cov(Bx.T)) / 2 + ridge * torch.eye(Dd, dtype=torch.double)
        w = torch.linalg.solve(cov, (Bx.mean(0) - A.mean(0)))
        aucs.append(_auc(Xs[te_s] @ w, Xt[te_t] @ w))
    return float(sum(aucs) / len(aucs)) if aucs else 0.5
 
 
# ----------------------------------------------------------------------------
# Collect descriptors
# ----------------------------------------------------------------------------
@torch.no_grad()
def collect(model, holder, loaders):
    """Gather per-image relative style spectra + style fraction for each domain."""
    profs = {d: [] for d in DOMAINS}
    sfrac = {d: [] for d in DOMAINS}
    for d in DOMAINS:
        seen = 0
        for bi, batch in enumerate(loaders[d]):
            if BATCHES_CAP and bi >= BATCHES_CAP:
                break
            model(batch)                                   # list[dict]; triggers CSFD pre-hook
            feat = holder['feat'].to(DEVICE).float()       # [B, C, H, W] entering CSFD
            prof, sf = style_spectrum(feat, BAND_SIGMAS)
            profs[d].append(prof.cpu()); sfrac[d].append(sf.cpu())
            seen += feat.shape[0]
            if seen >= IMAGES_PER_D:
                break
        profs[d] = torch.cat(profs[d], 0)[:IMAGES_PER_D]    # [N, K]
        sfrac[d] = torch.cat(sfrac[d], 0)[:IMAGES_PER_D]    # [N]
        print(f"  collected {seen:4d} imgs for {d}")
    return profs, sfrac
 
 
def main():
    model, holder, cfg = build_model_and_hook()
    loaders = domain_loaders(cfg)
 
    print("Collecting CSFD-input features per domain...")
    profs, sfrac = collect(model, holder, loaders)
 
    K = len(BAND_SIGMAS)
    band_names = [f"band{k}(s{BAND_SIGMAS[k]:g})" for k in range(K)]
    targets = [d for d in DOMAINS if d != SOURCE]
    mean_prof = {d: profs[d].mean(0) for d in DOMAINS}       # [K] each
 
    # --- 1. relative style-band spectrum per domain (scale-invariant) ---
    print("\n1) Relative style-band energy spectrum  (fractions sum to 1 across bands;"
          "\n   HIGH-freq -> LOW-freq left to right; global magnitude cancelled)")
    print("  {:12s}".format("") + "".join(f"{b:>12s}" for b in band_names)
          + f"{'style_frac':>12s}")
    for d in DOMAINS:
        row = "".join(f"{mean_prof[d][k]:12.3f}" for k in range(K))
        print(f"  {d:12s}" + row + f"{sfrac[d].mean():12.3f}")
 
    # --- 2. spectrum shift vs source, per target ---
    print("\n2) Spectrum shift vs {} (target - source);  + = target has relatively"
          "\n   MORE style energy in that band".format(SOURCE))
    peak_shift = {}
    for t in targets:
        gap = mean_prof[t] - mean_prof[SOURCE]              # [K]
        k_star = int(gap.abs().argmax())
        peak_shift[t] = k_star
        row = "".join(f"{gap[k]:+12.3f}" for k in range(K))
        print(f"  {t:12s}" + row + f"   most-shifted: {band_names[k_star]}")
 
    # --- 3. de-confounded separability: AUC on the low-dim spectrum only ---
    print("\n3) Held-out AUC on the {}-D relative spectrum (scale-invariant)."
          "\n   ~0.5 = target spectrum indistinguishable from source; ->1.0 = distinct"
          " spectral shape".format(K))
    for t in targets:
        print(f"  {t:12s} AUC = {lda_auc(profs[SOURCE], profs[t]):.3f}")
 
    # --- verdict: do TARGETS want DIFFERENT cutoffs from each other? ---
    print("\nVERDICT:")
    distinct = len(set(peak_shift.values()))
    max_gap = max((mean_prof[t] - mean_prof[SOURCE]).abs().max().item() for t in targets)
    if max_gap < 0.03:
        print(f"  All target spectra ~ source (max shift {max_gap:.3f} < 0.03).")
        print("  The content/style split sits at the same frequency everywhere ->")
        print("  a single fixed cutoff is adequate. Multi-band unlikely to help; skip Idea 4.")
    elif distinct == 1:
        print(f"  Targets differ from source (max shift {max_gap:.3f}) but ALL shift toward")
        print(f"  the SAME band ({band_names[list(peak_shift.values())[0]]}).")
        print("  One BETTER-PLACED fixed cutoff may help all targets; per-domain adaptivity")
        print("  (multi-band) is only weakly motivated. Consider just retuning init_sigma first.")
    else:
        print(f"  Targets shift toward DIFFERENT bands ({', '.join(f'{t}->{band_names[peak_shift[t]]}' for t in targets)}).")
        print("  No single cutoff serves all targets -> multi-band (per-band gate) is justified.")
        print("  Place the module's band_sigmas so bands straddle these peaks.")
 
    print("\nNOTE: this probes energy at the CSFD INPUT, which mixes content and style;")
    print("it bounds the premise, it does not prove the trained gate will exploit it.")
 
 
if __name__ == "__main__":
    main()

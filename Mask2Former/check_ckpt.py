import torch
c = torch.load("experiments/training_outputs/csfd_swinb_90k/model_final.pth", map_location="cpu")
v = torch.load("experiments/training_outputs/vanilla_mask2former_swinb_90k_v2/model_final.pth", map_location="cpu")
print("csfd iteration:", c.get("iteration"), " | vanilla iteration:", v.get("iteration"))
cm, vm = c["model"], v["model"]
for k in cm:
    if "predictor" in k and k.endswith(".weight") and cm[k].dim() > 1:
        print(k, "csfd norm=%.3f" % cm[k].norm(), "vanilla norm=%.3f" % vm.get(k, torch.zeros(1)).norm())
        break
gk = [k for k in cm if "csfd.gate" in k and k.endswith(".weight")][-1]
print("gate last-conv norm in file:", cm[gk].norm().item(), "(0.0 = saved at init)")

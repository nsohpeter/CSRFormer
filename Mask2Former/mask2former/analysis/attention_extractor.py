"""
attention_extractor.py
Extract self-attention, cross-attention, and query embeddings from
Mask2Former's MultiScaleMaskedTransformerDecoder via forward hooks.
 
STORAGE STRATEGY (memory-efficient):
    Raw cross-attention is [B, H, Q, K] where K can be up to 32768.
    We reduce on capture to avoid accumulating large tensors:
 
        self_attention[i]:  [H, Q, Q]   kept full  (Q=100, ~800KB/layer)
        cross_attention[i]: [H, K]      mean over Q (~1MB/layer)
        query_features[i]:  [Q, D]      kept full  (~100KB/layer)
 
    Per-image .pkl ≈ 2MB.  500 images ≈ 1GB total.
 
Decoder structure (Mask2Former / CMFormer):
    decoder.transformer_self_attention_layers[i].self_attn       <- nn.MHA
    decoder.transformer_cross_attention_layers[i].multihead_attn <- nn.MHA
    Forward order per layer: cross_attn -> self_attn -> ffn
    Queries are seq-first [Q, B, D] throughout.
 
Usage:
    extractor = AttentionExtractor(model)
    extractor.register_hooks()
    extractor.extract_and_save(inp, image_id, output_dir)
    extractor.remove_hooks()
 
Output per image (.pkl):
    {
        "image_id":        int,
        "image_path":      str,
        "self_attention":  {layer_idx: np.ndarray [H, Q, Q]},   # 9 layers
        "cross_attention": {layer_idx: np.ndarray [H, K]},      # 9 layers, K varies by scale
        "query_features":  {layer_idx: np.ndarray [Q, D]},      # 9 layers
        "metadata":        {num_heads, embed_dim, num_queries,
                            num_layers_captured, cross_attn_K_per_layer}
    }
"""
 
import gc
import logging
import os
import pickle
from typing import Dict, Optional
 
import numpy as np
import torch
import torch.nn as nn
 
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Hook storage  (module-level, cleared before every image)
# ---------------------------------------------------------------------------
_hook_storage: Dict[str, Dict[int, torch.Tensor]] = {
    "self_attn":  {},
    "cross_attn": {},
}
_query_storage: Dict[int, torch.Tensor] = {}
 
 
def _clear_storage():
    _hook_storage["self_attn"].clear()
    _hook_storage["cross_attn"].clear()
    _query_storage.clear()
 
 
# ---------------------------------------------------------------------------
# need_weights patch — forces per-head weights out of nn.MHA
# ---------------------------------------------------------------------------
_ORIG_FWD = "_orig_forward"
 
 
def _patch_need_weights(mha: nn.MultiheadAttention):
    """Idempotent. Forces need_weights=True, average_attn_weights=False."""
    if hasattr(mha, _ORIG_FWD):
        return
    orig = mha.forward
 
    def patched(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False  # per-head: [B, H, Q, K]
        return orig(*args, **kwargs)
 
    mha.forward = patched
    setattr(mha, _ORIG_FWD, orig)
 
 
def _unpatch_need_weights(mha: nn.MultiheadAttention):
    if hasattr(mha, _ORIG_FWD):
        mha.forward = getattr(mha, _ORIG_FWD)
        delattr(mha, _ORIG_FWD)
 
 
# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------
def _make_self_attn_hook(layer_idx: int):
    """Captures [B, H, Q, Q] → stores [H, Q, Q] (squeeze batch)."""
    def hook_fn(module, input, output):
        _, attn_weights = output
        if attn_weights is not None:
            _hook_storage["self_attn"][layer_idx] = attn_weights[0].detach().cpu()
        else:
            logger.warning(f"[self_attn layer {layer_idx}] attn_weights is None")
    return hook_fn
 
 
def _make_cross_attn_hook(layer_idx: int):
    """
    Captures [B, H, Q, K] → reduces to [H, K] by mean over Q.
    Keeps the per-head distribution over pixel features for JSD.
    """
    def hook_fn(module, input, output):
        _, attn_weights = output
        if attn_weights is not None:
            # [B, H, Q, K] -> squeeze B -> [H, Q, K] -> mean over Q -> [H, K]
            reduced = attn_weights[0].mean(dim=1).detach().cpu()
            _hook_storage["cross_attn"][layer_idx] = reduced
        else:
            logger.warning(f"[cross_attn layer {layer_idx}] attn_weights is None")
    return hook_fn
 
 
def _make_query_hook(layer_idx: int):
    """
    Hook on the CrossAttentionLayer module itself.
    input[0] = tgt = queries entering this layer, seq-first [Q, B, D].
    Stores [Q, D] (squeeze batch).
    """
    def hook_fn(module, input, output):
        tgt = input[0]  # [Q, B, D]
        _query_storage[layer_idx] = tgt[:, 0, :].detach().cpu()
    return hook_fn
 
 
# ---------------------------------------------------------------------------
# AttentionExtractor
# ---------------------------------------------------------------------------
class AttentionExtractor:
 
    def __init__(self, model: nn.Module, device: torch.device = torch.device("cuda")):
        self.model  = model
        self.device = device
        self._hooks: list = []
 
        self.decoder   = self._find_decoder()
        self._sa_layers = self.decoder.transformer_self_attention_layers
        self._ca_layers = self.decoder.transformer_cross_attention_layers
 
        n = len(self._sa_layers)
        assert n == len(self._ca_layers), "SA and CA layer counts mismatch"
        logger.info(f"AttentionExtractor: found {n} decoder layers")
 
    # -----------------------------------------------------------------------
    # Decoder locator
    # -----------------------------------------------------------------------
    def _find_decoder(self) -> nn.Module:
        """
        Try known paths first, then fall back to a named_modules walk.
        Mask2Former:  model.sem_seg_head.predictor
        CMFormer:     model.sem_seg_head.predictor  (same)
        """
        candidate_getters = [
            lambda m: m.sem_seg_head.predictor,
            lambda m: m.sem_seg_head.transformer.decoder,
            lambda m: m.module.sem_seg_head.predictor,
        ]
        for getter in candidate_getters:
            try:
                dec = getter(self.model)
                assert hasattr(dec, "transformer_self_attention_layers")
                assert hasattr(dec, "transformer_cross_attention_layers")
                logger.info("Decoder found via direct path ✓")
                return dec
            except (AttributeError, AssertionError):
                continue
 
        # Fallback: walk all modules
        for name, mod in self.model.named_modules():
            if mod.__class__.__name__ == "MultiScaleMaskedTransformerDecoder":
                logger.info(f"Decoder found via named_modules walk at: {name}")
                return mod
 
        raise RuntimeError(
            "Could not locate MultiScaleMaskedTransformerDecoder. "
            "Run: [print(n) for n, _ in model.named_modules()] to inspect the model."
        )
 
    # -----------------------------------------------------------------------
    # Hook management
    # -----------------------------------------------------------------------
    def register_hooks(self):
        self.remove_hooks()
        for i in range(len(self._sa_layers)):
            # Self-attention — vanilla M2F uses .self_attn, CMFormer uses .self_attn_high
            sa_layer = self._sa_layers[i]
            if hasattr(sa_layer, "self_attn"):
                sa_mha = sa_layer.self_attn
            elif hasattr(sa_layer, "self_attn_high"):
                sa_mha = sa_layer.self_attn_high   # HiLo: hook the high-freq branch
            else:
                raise AttributeError(
                    f"SA layer {i} has neither .self_attn nor .self_attn_high. "
                    f"Attributes: {list(sa_layer._modules.keys())}"
                )
            _patch_need_weights(sa_mha)
            self._hooks.append(sa_mha.register_forward_hook(_make_self_attn_hook(i)))
 
            # Cross-attention (same structure for both M2F and CMFormer)
            ca_mha = self._ca_layers[i].multihead_attn
            _patch_need_weights(ca_mha)
            self._hooks.append(ca_mha.register_forward_hook(_make_cross_attn_hook(i)))
 
            # Query features (hook on the whole CA layer to capture incoming queries)
            self._hooks.append(self._ca_layers[i].register_forward_hook(_make_query_hook(i)))
 
        logger.info(f"Registered {len(self._hooks)} hooks ({len(self._sa_layers)} layers × 3)")
 
    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for i in range(len(self._sa_layers)):
            sa_layer = self._sa_layers[i]
            if hasattr(sa_layer, "self_attn"):
                _unpatch_need_weights(sa_layer.self_attn)
            elif hasattr(sa_layer, "self_attn_high"):
                _unpatch_need_weights(sa_layer.self_attn_high)
            _unpatch_need_weights(self._ca_layers[i].multihead_attn)
 
    # -----------------------------------------------------------------------
    # Extraction
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def extract_and_save(self, inp: Dict, image_id: int, output_dir: str) -> Optional[str]:
        """
        Run one forward pass, capture attention, save to disk, free memory.
        Call in a loop — no accumulation across images.
 
        Args:
            inp:        Detectron2-style input dict for a single image.
            image_id:   Integer identifier used in the output filename.
            output_dir: Directory to save attention_{image_id}.pkl.
 
        Returns:
            Path to the saved .pkl, or None if hooks did not fire.
        """
        self.model.eval()
        _clear_storage()
 
        self.model([inp])
 
        # Collect numpy arrays and free torch tensors immediately
        sa_dict = {k: v.numpy().copy() for k, v in _hook_storage["self_attn"].items()}
        _hook_storage["self_attn"].clear()
 
        ca_dict = {}
        ca_K    = {}
        for k, v in _hook_storage["cross_attn"].items():
            arr       = v.numpy().copy()
            ca_dict[k] = arr
            ca_K[k]    = arr.shape[1]
        _hook_storage["cross_attn"].clear()
 
        q_dict = {k: v.numpy().copy() for k, v in _query_storage.items()}
        _query_storage.clear()
 
        if not sa_dict:
            logger.error(f"Image {image_id}: no layers captured — hooks did not fire")
            return None
 
        result = {
            "image_id":        image_id,
            "image_path":      inp.get("image_path", ""),
            "self_attention":  sa_dict,
            "cross_attention": ca_dict,
            "query_features":  q_dict,
            "metadata": {
                "num_heads":               self.decoder.num_heads,
                "embed_dim":               self.decoder.query_feat.embedding_dim,
                "num_queries":             self.decoder.num_queries,
                "num_layers_captured":     len(sa_dict),
                "cross_attn_K_per_layer":  ca_K,
            },
        }
 
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"attention_{image_id}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
 
        del result, sa_dict, ca_dict, q_dict
        gc.collect()
 
        return out_path
 
    # -----------------------------------------------------------------------
    # Load utility
    # -----------------------------------------------------------------------
    @staticmethod
    def load_result(path: str) -> Dict:
        with open(path, "rb") as f:
            return pickle.load(f)
 
 
# ---------------------------------------------------------------------------
# Validation utility
# ---------------------------------------------------------------------------
def validate_extracted_attention(result: Dict) -> bool:
    """
    Print a shape report and return True if all shapes are as expected.
 
    Expected shapes (Swin-B, 9-layer decoder):
        self_attention[i]:  [8, 100, 100]
        cross_attention[i]: [8, K]   K ∈ {2048, 8192, 32768}
        query_features[i]:  [100, 256]
    """
    ok        = True
    num_heads = result["metadata"]["num_heads"]
    num_q     = result["metadata"]["num_queries"]
    embed_dim = result["metadata"]["embed_dim"]
 
    print("\n" + "=" * 55)
    print("  Attention Shape Validation")
    print("=" * 55)
 
    for i in sorted(result["self_attention"]):
        sa  = result["self_attention"][i]
        tag = "✓" if sa.shape == (num_heads, num_q, num_q) else "✗"
        if tag == "✗": ok = False
        print(f"  self_attn   layer {i}: {str(sa.shape):>20s}  {tag}")
 
    print()
    for i in sorted(result["cross_attention"]):
        ca  = result["cross_attention"][i]
        tag = "✓" if (ca.shape[0] == num_heads and ca.shape[1] > 0) else "✗"
        if tag == "✗": ok = False
        print(f"  cross_attn  layer {i}: {str(ca.shape):>20s}  {tag}  (K={ca.shape[1]})")
 
    print()
    for i in sorted(result["query_features"]):
        qf  = result["query_features"][i]
        tag = "✓" if qf.shape == (num_q, embed_dim) else "✗"
        if tag == "✗": ok = False
        print(f"  queries     layer {i}: {str(qf.shape):>20s}  {tag}")
 
    print("-" * 55)
    print(f"  Overall: {'PASSED ✓' if ok else 'FAILED ✗'}")
    print("=" * 55 + "\n")
    return ok
"""
attention_extractor.py
Phase 1, Task 1.1 — Extract self-attention, cross-attention, and query embeddings
from Mask2Former's MultiScaleMaskedTransformerDecoder.

STORAGE STRATEGY (v2 — memory-efficient):
    Raw cross-attention is [B, H, Q, K] where K can be up to 32768.
    At float32 that's ~100MB per layer — 500 images would be ~195GB total.
    We don't need full [Q, K] for JSD or entropy. We reduce on capture:

        self_attention[i]:  [H, Q, Q]   kept full  — Q=100, so only 800KB/layer
        cross_attention[i]: [H, K]      mean over Q — enough for per-head JSD/entropy
        query_features[i]:  [Q, D]      kept full  — 100x256 = 100KB/layer

    Per-image .pkl drops from ~398MB → ~2MB.  500 images → ~1GB total.

Written against the ACTUAL decoder source:
    - Layers in 3 separate ModuleLists:
        decoder.transformer_self_attention_layers[i].self_attn        <- nn.MHA
        decoder.transformer_cross_attention_layers[i].multihead_attn  <- nn.MHA
    - Forward order per layer: cross_attn -> self_attn -> ffn
    - forward_pre/forward_post both discard weights with [0], so we patch
      need_weights=True and capture via hooks before that indexing.
    - output (queries) is seq-first [Q, B, D] throughout.

Usage:
    from mask2former.analysis.attention_extractor import AttentionExtractor

    extractor = AttentionExtractor(model)
    extractor.register_hooks()
    results = extractor.extract(inputs, image_ids)
    extractor.remove_hooks()
    AttentionExtractor.save_results(results, output_dir)

Output per image (.pkl):
    {
        "image_id":        int,
        "image_path":      str,
        "self_attention":  {layer_idx: np.array [H, Q, Q]},    # 9 layers
        "cross_attention": {layer_idx: np.array [H, K]},       # 9 layers, K varies by scale
        "query_features":  {layer_idx: np.array [Q, D]},       # 9 layers
        "metadata":        { num_heads, embed_dim, num_queries, num_layers_captured,
                             cross_attn_K_per_layer }
    }
"""

import os
import gc
import pickle
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  Hook-internal storage  (module-level, cleared per image)
# ---------------------------------------------------------------------------
_hook_storage: Dict[str, Dict[int, torch.Tensor]] = {
    "self_attn": {},
    "cross_attn": {},
}
_query_storage: Dict[int, torch.Tensor] = {}


def _clear_storage():
    _hook_storage["self_attn"].clear()
    _hook_storage["cross_attn"].clear()
    _query_storage.clear()


# ---------------------------------------------------------------------------
# 2.  need_weights patch
# ---------------------------------------------------------------------------
_ORIG_FWD = "_orig_forward"


def _patch_need_weights(mha: nn.MultiheadAttention):
    """Idempotent. Forces need_weights=True, average_attn_weights=False."""
    if hasattr(mha, _ORIG_FWD):
        return
    orig = mha.forward

    def patched(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False   # per-head: [B, H, Q, K]
        return orig(*args, **kwargs)

    mha.forward = patched
    setattr(mha, _ORIG_FWD, orig)


def _unpatch_need_weights(mha: nn.MultiheadAttention):
    if hasattr(mha, _ORIG_FWD):
        mha.forward = getattr(mha, _ORIG_FWD)
        delattr(mha, _ORIG_FWD)


# ---------------------------------------------------------------------------
# 3.  Hook factories
# ---------------------------------------------------------------------------
def _make_self_attn_hook(layer_idx: int):
    """
    Self-attention: output weights [B, H, Q, Q].
    We keep full [H, Q, Q] (squeeze B=1).  ~800KB per layer — fine.
    """
    def hook_fn(module, input, output):
        _, attn_weights = output          # [B, H, Q, Q]
        if attn_weights is not None:
            # squeeze batch dim, move to CPU, store
            _hook_storage["self_attn"][layer_idx] = attn_weights[0].detach().cpu()
        else:
            logger.warning(f"[self_attn layer {layer_idx}] attn_weights is None")
    return hook_fn


def _make_cross_attn_hook(layer_idx: int):
    """
    Cross-attention: output weights [B, H, Q, K], K up to 32768.
    Reduce immediately: mean over Q → [H, K].  This is the per-head
    attention distribution over pixel features — exactly what JSD and
    entropy operate on.  Drops ~100MB layers to ~1MB.
    """
    def hook_fn(module, input, output):
        _, attn_weights = output          # [B, H, Q, K]
        if attn_weights is not None:
            # [B, H, Q, K] -> mean over Q -> [B, H, K] -> squeeze B -> [H, K]
            reduced = attn_weights[0].mean(dim=1).detach().cpu()   # [H, K]
            _hook_storage["cross_attn"][layer_idx] = reduced
        else:
            logger.warning(f"[cross_attn layer {layer_idx}] attn_weights is None")
    return hook_fn


def _make_query_hook(layer_idx: int):
    """
    Hook on the full CrossAttentionLayer.  input[0] = tgt = queries entering
    this layer, seq-first [Q, B, D].  We store [Q, D] (squeeze B=1).
    """
    def hook_fn(module, input, output):
        tgt = input[0]                                     # [Q, B, D]
        _query_storage[layer_idx] = tgt[:, 0, :].detach().cpu()   # [Q, D]
    return hook_fn


# ---------------------------------------------------------------------------
# 4.  AttentionExtractor
# ---------------------------------------------------------------------------
class AttentionExtractor:

    def __init__(self, model: nn.Module, device: torch.device = torch.device("cuda")):
        self.model = model
        self.device = device
        self._hooks: list = []

        self.decoder = self._find_decoder()
        self._sa_layers = self.decoder.transformer_self_attention_layers
        self._ca_layers = self.decoder.transformer_cross_attention_layers

        n = len(self._sa_layers)
        assert n == len(self._ca_layers)
        logger.info(f"AttentionExtractor: found {n} decoder layers")

    # --- locate decoder ----------------------------------------------------
    def _find_decoder(self) -> nn.Module:
        paths = [
            lambda m: m.sem_seg_head.transformer.decoder,
            lambda m: m.sem_seg_head.transformer,
            lambda m: m.module.sem_seg_head.transformer.decoder,
        ]
        for getter in paths:
            try:
                dec = getter(self.model)
                assert hasattr(dec, "transformer_self_attention_layers")
                assert hasattr(dec, "transformer_cross_attention_layers")
                logger.info("Decoder found ✓")
                return dec
            except (AttributeError, AssertionError):
                continue

        for name, mod in self.model.named_modules():
            if mod.__class__.__name__ == "MultiScaleMaskedTransformerDecoder":
                logger.info(f"Decoder found via named_modules walk at: {name}")
                return mod

        raise RuntimeError(
            "Could not locate MultiScaleMaskedTransformerDecoder.\n"
            "Run: for name, mod in model.named_modules(): print(name, type(mod))"
        )

    # --- hook management ---------------------------------------------------
    def register_hooks(self):
        self.remove_hooks()

        for i in range(len(self._sa_layers)):
            sa_mha = self._sa_layers[i].self_attn
            _patch_need_weights(sa_mha)
            self._hooks.append(
                sa_mha.register_forward_hook(_make_self_attn_hook(i))
            )

            ca_mha = self._ca_layers[i].multihead_attn
            _patch_need_weights(ca_mha)
            self._hooks.append(
                ca_mha.register_forward_hook(_make_cross_attn_hook(i))
            )

            self._hooks.append(
                self._ca_layers[i].register_forward_hook(_make_query_hook(i))
            )

        logger.info(f"Registered {len(self._hooks)} hooks ({len(self._sa_layers)} layers x 3)")

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for i in range(len(self._sa_layers)):
            _unpatch_need_weights(self._sa_layers[i].self_attn)
            _unpatch_need_weights(self._ca_layers[i].multihead_attn)

    # --- extraction --------------------------------------------------------
    @torch.no_grad()
    def extract_and_save(self, inp: Dict, image_id: int, output_dir: str):
        """
        Run one image, capture attention, save to disk, free memory.
        No accumulation — this is called in a loop by the runner.
        """
        self.model.eval()
        _clear_storage()

        _ = self.model([inp])

        # --- collect into numpy, free torch tensors immediately ------------
        sa_dict = {}
        for k, v in _hook_storage["self_attn"].items():
            sa_dict[k] = v.numpy().copy()
        _hook_storage["self_attn"].clear()

        ca_dict = {}
        ca_K = {}
        for k, v in _hook_storage["cross_attn"].items():
            arr = v.numpy().copy()
            ca_dict[k] = arr
            ca_K[k] = arr.shape[1]      # K dimension per layer
        _hook_storage["cross_attn"].clear()

        q_dict = {}
        for k, v in _query_storage.items():
            q_dict[k] = v.numpy().copy()
        _query_storage.clear()

        num_layers = len(sa_dict)
        if num_layers == 0:
            logger.error(f"Image {image_id}: zero layers captured — hooks did not fire")
            return

        # --- build result & save -------------------------------------------
        result = {
            "image_id":        image_id,
            "image_path":      inp.get("image_path", ""),
            "self_attention":  sa_dict,
            "cross_attention": ca_dict,
            "query_features":  q_dict,
            "metadata": {
                "num_heads":            self.decoder.num_heads,
                "embed_dim":            self.decoder.query_feat.embedding_dim,
                "num_queries":          self.decoder.num_queries,
                "num_layers_captured":  num_layers,
                "cross_attn_K_per_layer": ca_K,
            },
        }

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"attention_{image_id}.pkl")
        with open(path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

        # free everything
        del result, sa_dict, ca_dict, q_dict
        gc.collect()

        return path

    # --- legacy batch extract (kept for compatibility) ---------------------
    @torch.no_grad()
    def extract(self, inputs: List[Dict], image_ids: Optional[List[int]] = None) -> List[Dict]:
        self.model.eval()
        results = []
        for i, inp in enumerate(inputs):
            _clear_storage()
            _ = self.model([inp])

            sa_dict  = {k: v.numpy() for k, v in sorted(_hook_storage["self_attn"].items())}
            ca_dict  = {k: v.numpy() for k, v in sorted(_hook_storage["cross_attn"].items())}
            q_dict   = {k: v.numpy() for k, v in sorted(_query_storage.items())}

            results.append({
                "image_id":        image_ids[i] if image_ids else i,
                "image_path":      inp.get("image_path", ""),
                "self_attention":  sa_dict,
                "cross_attention": ca_dict,
                "query_features":  q_dict,
                "metadata": {
                    "num_heads":            self.decoder.num_heads,
                    "embed_dim":            self.decoder.query_feat.embedding_dim,
                    "num_queries":          self.decoder.num_queries,
                    "num_layers_captured":  len(sa_dict),
                    "cross_attn_K_per_layer": {k: v.shape[1] for k, v in ca_dict.items()},
                },
            })
        return results

    # --- I/O ---------------------------------------------------------------
    @staticmethod
    def save_results(results: List[Dict], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for res in results:
            path = os.path.join(output_dir, f"attention_{res['image_id']}.pkl")
            with open(path, "wb") as f:
                pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved -> {path}")

    @staticmethod
    def load_result(path: str) -> Dict:
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# 5.  Validation utility
# ---------------------------------------------------------------------------
def validate_extracted_attention(result: Dict) -> bool:
    """
    Shape report for the compressed format.

    Expected:
        self_attention[i]:  [H, Q, Q]   = [8, 100, 100]
        cross_attention[i]: [H, K]      = [8, K]  where K in {2048, 8192, 32768}
        query_features[i]:  [Q, D]      = [100, 256]
    """
    ok = True
    num_heads = result["metadata"]["num_heads"]       # 8
    num_q     = result["metadata"]["num_queries"]     # 100
    embed_dim = result["metadata"]["embed_dim"]       # 256

    print("\n" + "=" * 55)
    print("  Attention Shape Validation (compressed format)")
    print("=" * 55)

    for layer_idx in sorted(result["self_attention"].keys()):
        sa = result["self_attention"][layer_idx]
        tag = "✓" if sa.shape == (num_heads, num_q, num_q) else "✗"
        if tag == "✗": ok = False
        print(f"  self_attn   layer {layer_idx}: {str(sa.shape):>20s}  {tag}")

    print()
    for layer_idx in sorted(result["cross_attention"].keys()):
        ca = result["cross_attention"][layer_idx]
        tag = "✓" if (ca.shape[0] == num_heads and ca.shape[1] > 0) else "✗"
        if tag == "✗": ok = False
        print(f"  cross_attn  layer {layer_idx}: {str(ca.shape):>20s}  {tag}   (K={ca.shape[1]})")

    print()
    for layer_idx in sorted(result["query_features"].keys()):
        qf = result["query_features"][layer_idx]
        tag = "✓" if qf.shape == (num_q, embed_dim) else "✗"
        if tag == "✗": ok = False
        print(f"  queries     layer {layer_idx}: {str(qf.shape):>20s}  {tag}")

    print("-" * 55)
    print(f"  Overall: {'PASSED ✓' if ok else 'FAILED ✗'}")
    print("=" * 55 + "\n")
    return ok

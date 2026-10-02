"""
aca_loss.py — Assignment-Aware Cross-View Consistency Loss
 
Motivation:
    Mask2Former queries are unordered. Naive attention consistency
    (JSD between A_orig[i] and A_aug[i]) compares semantically different
    queries ~85% of the time on real images (measured on Cityscapes val).
 
    ACA fixes this by first finding the optimal query correspondence via
    Hungarian matching on mask predictions, then enforcing JSD consistency
    on the correctly aligned pairs:
 
        JSD(A_orig[ℓ][i], A_aug[ℓ][π(i)])
 
    where π is the permutation mapping original query i to the augmented
    query that predicts the same segment.
 
Loss:
    L_aca = λ_attn * (1/|L|) Σ_{ℓ∈L} JSD(A_orig[ℓ], A_aug[ℓ][π])
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, Optional
import numpy as np
 
EPS = 1e-8
 
 
# ---------------------------------------------------------------------------
# Hungarian matching
# ---------------------------------------------------------------------------
 
@torch.no_grad()
def hungarian_match_masks(
    masks_orig: torch.Tensor,   # [B, Q, H, W]
    masks_aug:  torch.Tensor,   # [B, Q, H, W]
) -> torch.Tensor:
    """
    Find optimal query correspondence between two mask sets using
    Hungarian algorithm on pairwise Dice cost.
 
    Returns:
        perm: [B, Q]  — perm[b, i] = j means aug query j corresponds
                        to orig query i for batch item b.
    """
    B, Q, H, W = masks_orig.shape
    device = masks_orig.device
 
    # Sigmoid + flatten for Dice computation
    m_o = masks_orig.sigmoid().flatten(2)   # [B, Q, H*W]
    m_a = masks_aug.sigmoid().flatten(2)    # [B, Q, H*W]
 
    # Pairwise Dice cost: [B, Q, Q]
    inter  = torch.bmm(m_o, m_a.transpose(1, 2))         # [B, Q, Q]
    sum_o  = m_o.sum(2, keepdim=True)                     # [B, Q, 1]
    sum_a  = m_a.sum(2).unsqueeze(1)                      # [B, 1, Q]
    dice_cost = 1.0 - (2.0 * inter + EPS) / (sum_o + sum_a + EPS)
 
    # Run Hungarian per batch item (on CPU via scipy)
    cost_np = dice_cost.cpu().numpy()
    perm    = torch.zeros(B, Q, dtype=torch.long, device=device)
 
    for b in range(B):
        cost_b = cost_np[b]
        # Guard against NaN/Inf from extreme mask values during early training
        if not np.isfinite(cost_b).all():
            cost_b = np.nan_to_num(cost_b, nan=1.0, posinf=1.0, neginf=0.0)
        _, col_ind = linear_sum_assignment(cost_b)
        perm[b] = torch.from_numpy(col_ind).to(device)
 
    return perm   # perm[b, i] = j  (aug query j ↔ orig query i)
 
 
# ---------------------------------------------------------------------------
# JSD
# ---------------------------------------------------------------------------
 
def jsd_loss(
    p: torch.Tensor,   # [B, H, Q, K]
    q: torch.Tensor,   # [B, H, Q, K]
) -> torch.Tensor:
    """Jensen-Shannon Divergence, mean over B, H, Q."""
    p = p.clamp(min=EPS)
    q = q.clamp(min=EPS)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)   # [B, H, Q]
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return (0.5 * (kl_pm + kl_qm)).mean()
 
 
# ---------------------------------------------------------------------------
# ACALoss
# ---------------------------------------------------------------------------
 
class ACALoss(nn.Module):
    """
    Assignment-Aware Cross-View Consistency Loss.
 
    Args:
        lambda_attn:  Weight for attention JSD loss       (default: 0.1)
        lambda_query: Weight for query L2 loss            (default: 0.0)
        max_layer:    Only apply to layers 0..max_layer-1 (default: 5)
    """
 
    def __init__(
        self,
        lambda_attn:  float = 0.1,
        lambda_query: float = 0.0,
        max_layer:    Optional[int] = 5,
    ):
        super().__init__()
        self.lambda_attn  = lambda_attn
        self.lambda_query = lambda_query
        self.max_layer    = max_layer
 
    def forward(
        self,
        attn_orig:   Dict[int, torch.Tensor],   # {ℓ: [B, H, Q, K]}
        attn_aug:    Dict[int, torch.Tensor],   # {ℓ: [B, H, Q, K]}
        query_orig:  Dict[int, torch.Tensor],   # {ℓ: [B, Q, D]}
        query_aug:   Dict[int, torch.Tensor],   # {ℓ: [B, Q, D]}
        masks_orig:  torch.Tensor,              # [B, Q, H, W]
        masks_aug:   torch.Tensor,              # [B, Q, H, W]
    ) -> Dict[str, torch.Tensor]:
 
        # Step 1 — Hungarian matching (no grad, on CPU)
        perm = hungarian_match_masks(
            masks_orig.detach(),
            masks_aug.detach(),
        )   # [B, Q]
 
        # Step 2 — Select layers
        all_layers = sorted(attn_orig.keys())
        layers = (
            [l for l in all_layers if l < self.max_layer]
            if self.max_layer is not None
            else all_layers
        )
 
        attn_losses  = []
        query_losses = []
 
        for layer_idx in layers:
            if layer_idx not in attn_aug:
                continue
 
            A_orig = attn_orig[layer_idx]          # [B, H, Q, K]
            A_aug  = attn_aug[layer_idx].detach()  # stop-grad
 
            # Step 3 — Re-index augmented attention by permutation
            # A_aug_aligned[b, h, i, :] = A_aug[b, h, perm[b,i], :]
            B, H, Q, K = A_aug.shape
            perm_exp   = perm.unsqueeze(1).unsqueeze(-1).expand(B, H, Q, K)
            A_aug_perm = torch.gather(A_aug, dim=2, index=perm_exp)
 
            # Step 4 — JSD on aligned pairs
            attn_losses.append(jsd_loss(A_orig, A_aug_perm))
 
            # Optional query L2
            if self.lambda_query > 0 and layer_idx in query_aug:
                Q_orig = query_orig[layer_idx]          # [B, Q, D]
                Q_aug  = query_aug[layer_idx].detach()
 
                D = Q_aug.shape[-1]
                perm_q = perm.unsqueeze(-1).expand(B, Q, D)
                Q_aug_perm = torch.gather(Q_aug, dim=1, index=perm_q)
                query_losses.append(
                    F.mse_loss(Q_orig, Q_aug_perm)
                )
 
        # Aggregate
        zero = attn_orig[all_layers[0]].sum() * 0.0
 
        loss_attn  = torch.stack(attn_losses).mean()  if attn_losses  else zero
        loss_query = torch.stack(query_losses).mean() if query_losses else zero.detach()
 
        loss_attn_w  = self.lambda_attn  * loss_attn
        loss_query_w = self.lambda_query * loss_query
 
        return {
            "loss_aca_attn":  loss_attn_w,
            "loss_aca_query": loss_query_w,
            "loss_aca":       loss_attn_w + loss_query_w,
        }
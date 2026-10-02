"""
Content-Style Feature Decomposition (CSFD) Module
==================================================

Spatial frequency-based content-style decomposition designed for
transformer-based mask-classification architectures (Mask2Former).

Key design principles (derived from diagnostic experiments):
1. Operates on spatial frequencies, NOT channel statistics
   (IN on mask_features destroys -51 mIoU because einsum depends on
   channel magnitudes — a property unique to mask-classification)
2. Inserted between pixel decoder and transformer decoder
   (per-layer diagnosis showed 82% of OOD gap exists at layer 0)
3. Preserves channel magnitude structure throughout
   (transformer features encode semantics in channels, unlike CNN texture bias)

Architecture:
    mask_features [B, C, H, W]
              │
        ┌─────┴─────┐
        │            │
    Low-pass     High-pass (residual)
    (content)    (style/texture)
        │            │
        │      Style Robustifier
        │            │
        └─────┬─────┘
              │
        Gated Fusion
              │
        cleaned mask_features [B, C, H, W]

Usage:
    from mask2former.modeling.csfd.csfd_module import CSFD

    csfd = CSFD(in_channels=256)
    cleaned = csfd(mask_features)  # [B, 256, H, W] -> [B, 256, H, W]

Training-time addition (v1 + consistency loss):
    from mask2former.modeling.csfd.csfd_module import ContentConsistencyLoss

    consistency_loss = ContentConsistencyLoss()
    loss = consistency_loss(content_clean, content_aug)  # scalar
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableGaussianLowPass(nn.Module):
    """Depthwise convolution initialized as Gaussian blur.
    
    Separates low-frequency spatial content (shapes, structure, boundaries)
    from high-frequency detail (texture, fine patterns).
    
    Each channel is filtered independently (depthwise), preserving
    the channel magnitude structure that the mask prediction einsum requires.
    """
    
    def __init__(self, channels, kernel_size=7, init_sigma=2.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Depthwise conv: each channel gets its own spatial filter
        self.conv = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=self.padding,
            groups=channels,  # depthwise = per-channel
            bias=False,
        )
        
        # Initialize as Gaussian
        self._init_gaussian(init_sigma)
    
    def _init_gaussian(self, sigma):
        """Initialize kernel weights as a 2D Gaussian."""
        k = self.kernel_size
        center = k // 2
        
        # Create 2D Gaussian kernel
        coords = torch.arange(k, dtype=torch.float32) - center
        g1d = torch.exp(-coords**2 / (2 * sigma**2))
        g2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)  # outer product
        g2d = g2d / g2d.sum()  # normalize to sum=1 (preserves magnitude)
        
        # Set all channels to the same Gaussian init
        with torch.no_grad():
            self.conv.weight.copy_(
                g2d.unsqueeze(0).unsqueeze(0).expand(self.channels, 1, k, k)
            )
    
    def forward(self, x):
        return self.conv(x)


class StyleRobustifier(nn.Module):
    """Makes high-frequency (texture/style) features domain-invariant.
    
    This is v1's design (fixed IN + learned residual transform), reinstated
    after v2 (fixed channel gate) and v3 (input-adaptive channel alpha)
    both underperformed v1 on the full OOD benchmark. v1 remains the
    strongest architecture; the BDD gap is now addressed via a training-time
    consistency loss (see ContentConsistencyLoss below) rather than further
    architectural changes to this module.
    
    Applies Instance Normalization to the high-frequency component ONLY.
    Safe because the high-freq component captures texture/detail, not
    semantic structure, and IN here strips domain-specific texture
    statistics without touching the content branch's channel magnitudes.
    """
    
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid_channels = max(channels // reduction, 64)
        
        self.norm = nn.InstanceNorm2d(channels, affine=True)
        
        self.transform = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
        )
        nn.init.zeros_(self.transform[-1].weight)
    
    def forward(self, style_features):
        normed = self.norm(style_features)
        return style_features + self.transform(normed)


class SpatialGate(nn.Module):
    """Spatially-aware gating for content-style fusion.
    
    Channel-level control is now handled by AdaptiveStyleRobustifier's
    per-channel alpha, so this gate stays spatial-only (WHERE to inject
    style), matching v1's design rather than v2's combined spatial+channel
    gate — stacking two channel-control mechanisms would make attribution
    of any gain/loss ambiguous.
    
    Boundaries -> gate low (rely on content, domain-invariant shapes).
    Homogeneous regions -> gate high (texture helps class disambiguation).
    """
    
    def __init__(self, channels):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1, bias=True),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, 1.0)
    
    def forward(self, content, style_robust):
        gate_input = torch.cat([content, style_robust], dim=1)
        gate = torch.sigmoid(self.gate_net(gate_input))  # [B, 1, H, W]
        return gate


class CSFD(nn.Module):
    """Content-Style Feature Decomposition module — v1 architecture
    (reinstated as the strongest performer after v2/v3 underperformed it).
    
    Decomposes pixel decoder features into low-frequency content and
    high-frequency style components via spatial frequency separation,
    robustifies the style component against domain shift, and fuses
    them back with a learned spatial gate.
    
    v1 -> v2 -> v3 -> v1+consistency-loss history:
        v1: fixed IN-strength style robustifier + spatial-only gate.
            Best overall OOD mean (63.92); modest BDD regression (-0.13).
        v2: v1 + learned channel gate (static policy). Fixed BDD (+0.20)
            but broke GTA5/Map (-2.35 / -1.30). Static policy didn't transfer.
        v3: v1 + input-adaptive channel alpha (predicted from whole-image
            global stats). Failed to fix BDD (57.28, worse than v1 and v2) —
            global per-image pooling washes out small-object signal (bus,
            rider, bicycle), so the predicted alpha isn't actually sensitive
            to what BDD needs.
        v1 + consistency loss (current): keep v1's architecture (proven
            strongest), address the BDD gap via a training-time
            content-consistency loss instead of further architecture
            changes. See ContentConsistencyLoss below.
    
    Args:
        in_channels: Number of input feature channels (256 for Mask2Former)
        kernel_size: Size of the learnable low-pass filter (default: 7)
        init_sigma: Initial Gaussian sigma for the low-pass filter (default: 2.0)
        reduction: Channel reduction ratio in the style robustifier (default: 4)
    """
    
    def __init__(self, in_channels=256, kernel_size=7, init_sigma=2.0, reduction=4):
        super().__init__()
        
        self.in_channels = in_channels
        
        self.low_pass = LearnableGaussianLowPass(
            channels=in_channels,
            kernel_size=kernel_size,
            init_sigma=init_sigma,
        )
        
        self.style_robustifier = StyleRobustifier(
            channels=in_channels,
            reduction=reduction,
        )
        
        self.gate = SpatialGate(channels=in_channels)
    
    def get_content(self, mask_features):
        """Return ONLY the low-frequency content branch output.
        
        Used by the training-time content-consistency loss, which needs
        to compare content extracted from a clean image against content
        extracted from a photometrically-augmented version of the same
        image — without paying for the full style-robustify + gate path
        on the augmented view.
        """
        return self.low_pass(mask_features)
    
    def forward(self, mask_features):
        """
        Args:
            mask_features: [B, C, H, W] from pixel decoder
        Returns:
            cleaned_features: [B, C, H, W] for transformer decoder
        """
        content = self.low_pass(mask_features)
        style = mask_features - content
        
        style_robust = self.style_robustifier(style)
        
        gate = self.gate(content, style_robust)  # [B, 1, H, W]
        
        cleaned = content + gate * style_robust
        
        return cleaned
    
    def extra_repr(self):
        return (
            f"in_channels={self.in_channels}, "
            f"kernel_size={self.low_pass.kernel_size}, "
            f"params={sum(p.numel() for p in self.parameters()) / 1e3:.1f}K"
        )


class ContentConsistencyLoss(nn.Module):
    """Training-time content-consistency loss for CSFD (v1 + consistency-loss).
    
    Motivation: three independent architectural diagnoses (from other tools)
    all converged on "add global illumination context to the gate" to fix
    BDD — functionally identical to v3, already tried and already failed
    (57.28, worse than v1). Their premise (night/rain "corrupt frequencies")
    is also directly contradicted by CSFD's own results: the best deltas are
    on ACDC-Night (+2.27) and ACDC-Snow (+4.75), not the worst, ruling out
    an illumination-severity explanation.
    
    This is the other, untried idea those diagnoses converged on: rather
    than changing the architecture again, add a training-time regularizer
    that forces the content branch (low-pass output) to be stable under a
    severe photometric perturbation — i.e. explicitly teach content
    extraction to ignore exactly the kind of appearance shift the OOD
    targets exhibit, without touching v1's proven architecture.
    
    Literature grounding: SHADE-style consistency regularization; DSU
    (Li et al., ICLR 2022) as a citable fallback if this alone doesn't
    close the BDD gap.
    
    Compares content per spatial location, across channels — matching
    CSFD's own design constraint that channels (not spatial position)
    carry the magnitude structure the mask-prediction einsum depends on,
    so channel-wise vectors are the right unit for a cosine comparison.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, content_clean, content_aug):
        """
        Args:
            content_clean: [B, C, H, W] content branch output on the clean image
            content_aug:   [B, C, H, W] content branch output on the
                photometrically-augmented view of the SAME image
        Returns:
            scalar loss: 0 when perfectly aligned, up to 2 when perfectly
            anti-aligned (1 - cosine_similarity, averaged over all pixels)
        """
        b, c, h, w = content_clean.shape
        
        # Reshape to [B*H*W, C] so cosine similarity is computed per pixel,
        # across the channel dimension (channels carry semantic content here)
        clean_flat = content_clean.permute(0, 2, 3, 1).reshape(-1, c)
        aug_flat = content_aug.permute(0, 2, 3, 1).reshape(-1, c)
        
        cos_sim = F.cosine_similarity(clean_flat, aug_flat, dim=1)
        return 1.0 - cos_sim.mean()


def build_csfd(cfg):
    """Build CSFD module from detectron2 config."""
    return CSFD(
        in_channels=cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,  # 256 for Mask2Former
        kernel_size=cfg.MODEL.CSFD.KERNEL_SIZE if hasattr(cfg.MODEL, 'CSFD') else 7,
        init_sigma=cfg.MODEL.CSFD.INIT_SIGMA if hasattr(cfg.MODEL, 'CSFD') else 2.0,
        reduction=cfg.MODEL.CSFD.REDUCTION if hasattr(cfg.MODEL, 'CSFD') else 4,
    )


# ============================================================================
# Quick test / parameter count
# ============================================================================
if __name__ == "__main__":
    # Test with Mask2Former's default mask_features shape
    csfd = CSFD(in_channels=256)
    
    # Count parameters
    total_params = sum(p.numel() for p in csfd.parameters())
    trainable_params = sum(p.numel() for p in csfd.parameters() if p.requires_grad)
    print(f"CSFD Module (v1 architecture + consistency loss ready):")
    print(f"  Total parameters:     {total_params:,} ({total_params/1e3:.1f}K)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e3:.1f}K)")
    print(f"  Module structure:")
    print(csfd)
    
    # Test forward pass
    x = torch.randn(2, 256, 128, 256)  # typical Cityscapes mask_features size
    y = csfd(x)
    print(f"\n  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    
    # Verify near-identity initialization
    diff = (y - x).abs().mean().item()
    print(f"  Mean absolute diff from input (should be small): {diff:.6f}")
    
    # Verify channel magnitudes are preserved
    in_chan_mag = x.mean(dim=[0, 2, 3])  # [C]
    out_chan_mag = y.mean(dim=[0, 2, 3])
    mag_corr = torch.corrcoef(torch.stack([in_chan_mag, out_chan_mag]))[0, 1].item()
    print(f"  Channel magnitude correlation (should be ~1.0): {mag_corr:.6f}")
    
    # Test get_content() helper (used by the consistency loss)
    content_only = csfd.get_content(x)
    print(f"  get_content() output shape: {content_only.shape}")
    
    # Test ContentConsistencyLoss
    consistency_loss = ContentConsistencyLoss()
    content_a = csfd.get_content(x)
    content_b = csfd.get_content(x + torch.randn_like(x) * 0.1)  # mild perturbation
    loss_val = consistency_loss(content_a, content_b)
    print(f"  ContentConsistencyLoss on mildly perturbed input: {loss_val.item():.6f}")
    identical_loss = consistency_loss(content_a, content_a)
    print(f"  ContentConsistencyLoss on identical input (should be ~0): {identical_loss.item():.6f}")
 


# """
# Content-Style Feature Decomposition (CSFD) Module
# ==================================================
 
# Spatial frequency-based content-style decomposition designed for
# transformer-based mask-classification architectures (Mask2Former).
 
# Key design principles (derived from diagnostic experiments):
# 1. Operates on spatial frequencies, NOT channel statistics
#    (IN on mask_features destroys -51 mIoU because einsum depends on
#    channel magnitudes — a property unique to mask-classification)
# 2. Inserted between pixel decoder and transformer decoder
#    (per-layer diagnosis showed 82% of OOD gap exists at layer 0)
# 3. Preserves channel magnitude structure throughout
#    (transformer features encode semantics in channels, unlike CNN texture bias)
 
# Architecture:
#     mask_features [B, C, H, W]
#               │
#         ┌─────┴─────┐
#         │            │
#     Low-pass     High-pass (residual)
#     (content)    (style/texture)
#         │            │
#         │      Style Robustifier
#         │            │
#         └─────┬─────┘
#               │
#         Gated Fusion
#               │
#         cleaned mask_features [B, C, H, W]
 
# Usage:
#     from mask2former.modeling.csfd.csfd_module import CSFD
    
#     csfd = CSFD(in_channels=256)
#     cleaned = csfd(mask_features)  # [B, 256, H, W] -> [B, 256, H, W]
# """
 
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
 
 
# class LearnableGaussianLowPass(nn.Module):
#     """Depthwise convolution initialized as Gaussian blur.
    
#     Separates low-frequency spatial content (shapes, structure, boundaries)
#     from high-frequency detail (texture, fine patterns).
    
#     Each channel is filtered independently (depthwise), preserving
#     the channel magnitude structure that the mask prediction einsum requires.
#     """
    
#     def __init__(self, channels, kernel_size=7, init_sigma=2.0):
#         super().__init__()
#         self.channels = channels
#         self.kernel_size = kernel_size
#         self.padding = kernel_size // 2
        
#         # Depthwise conv: each channel gets its own spatial filter
#         self.conv = nn.Conv2d(
#             channels, channels,
#             kernel_size=kernel_size,
#             padding=self.padding,
#             groups=channels,  # depthwise = per-channel
#             bias=False,
#         )
        
#         # Initialize as Gaussian
#         self._init_gaussian(init_sigma)
    
#     def _init_gaussian(self, sigma):
#         """Initialize kernel weights as a 2D Gaussian."""
#         k = self.kernel_size
#         center = k // 2
        
#         # Create 2D Gaussian kernel
#         coords = torch.arange(k, dtype=torch.float32) - center
#         g1d = torch.exp(-coords**2 / (2 * sigma**2))
#         g2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)  # outer product
#         g2d = g2d / g2d.sum()  # normalize to sum=1 (preserves magnitude)
        
#         # Set all channels to the same Gaussian init
#         with torch.no_grad():
#             self.conv.weight.copy_(
#                 g2d.unsqueeze(0).unsqueeze(0).expand(self.channels, 1, k, k)
#             )
    
#     def forward(self, x):
#         return self.conv(x)
 
 
# class StyleRobustifier(nn.Module):
#     """Makes high-frequency (texture/style) features domain-invariant.
    
#     Applies Instance Normalization to the high-frequency component ONLY.
#     This is safe because:
#     - The high-freq component captures texture/detail, not semantic structure
#     - IN here strips domain-specific texture statistics without touching
#       the content branch's channel magnitudes
#     - A learned residual path preserves useful texture information
    
#     Architecture: IN → 1x1 Conv → ReLU → 1x1 Conv (zero-init) + residual
#     """
    
#     def __init__(self, channels, reduction=4):
#         super().__init__()
#         mid_channels = max(channels // reduction, 64)
        
#         # Instance Normalization with learnable affine
#         # Strips domain-specific texture statistics
#         self.norm = nn.InstanceNorm2d(channels, affine=True)
        
#         # Lightweight transform to learn domain-invariant texture representation
#         self.transform = nn.Sequential(
#             nn.Conv2d(channels, mid_channels, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, channels, 1, bias=False),
#         )
        
#         # Zero-initialize the last conv so the residual starts as identity
#         nn.init.zeros_(self.transform[-1].weight)
    
#     def forward(self, style_features):
#         # Normalize texture statistics (safe on high-freq component only)
#         normed = self.norm(style_features)
        
#         # Learn domain-invariant transformation + residual
#         # At init: transform outputs zeros, so output = style_features (identity)
#         return style_features + self.transform(normed)
 
 
# class SpatioChannelGate(nn.Module):
#     """Spatially and channel-aware gating for content-style fusion.
    
#     Produces a combined gate [B, C, H, W] from two components:
    
#     Spatial gate [B, 1, H, W]:
#         WHERE to inject style. Boundaries → low (rely on content).
#         Homogeneous regions → high (texture helps disambiguation).
    
#     Channel gate [B, C, 1, 1]:
#         WHICH channels of style to keep. Channels with useful texture
#         (road surface, vegetation pattern) → high. Channels with
#         domain-specific artifacts (illumination, camera bias) → low.
    
#     Combined: gate = spatial_gate × channel_gate → [B, C, H, W]
    
#     This addresses the BDD limitation: BDD's multi-camera diversity
#     embeds discriminative texture in specific channels. The channel
#     gate learns to preserve those while suppressing domain-specific ones.
#     """
    
#     def __init__(self, channels, reduction=16):
#         super().__init__()
        
#         # Spatial gate: WHERE to inject style [B, 1, H, W]
#         self.spatial_gate = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, 1, 1, bias=True),
#         )
        
#         # Initialize spatial gate bias to +1 (near pass-through)
#         nn.init.zeros_(self.spatial_gate[-1].weight)
#         nn.init.constant_(self.spatial_gate[-1].bias, 1.0)
        
#         # Channel gate: WHICH channels to preserve [B, C, 1, 1]
#         # Squeeze-and-excitation style: GAP → MLP → sigmoid
#         mid = max(channels // reduction, 16)
#         self.channel_gate = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),              # [B, 2C, 1, 1]
#             nn.Flatten(1),                         # [B, 2C]
#             nn.Linear(channels * 2, mid, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(mid, channels, bias=True),   # [B, C]
#         )
        
#         # Initialize channel gate bias to +1 (near pass-through)
#         nn.init.zeros_(self.channel_gate[-1].weight)
#         nn.init.constant_(self.channel_gate[-1].bias, 1.0)
    
#     def forward(self, content, style_robust):
#         gate_input = torch.cat([content, style_robust], dim=1)  # [B, 2C, H, W]
        
#         # Spatial: where
#         spatial = torch.sigmoid(self.spatial_gate(gate_input))  # [B, 1, H, W]
        
#         # Channel: which
#         channel = torch.sigmoid(
#             self.channel_gate(gate_input).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
#         )
        
#         # Combined gate [B, C, H, W]
#         gate = spatial * channel
#         return gate
 
 
# class CSFD(nn.Module):
#     """Content-Style Feature Decomposition module.
    
#     Decomposes pixel decoder features into low-frequency content and
#     high-frequency style components via spatial frequency separation,
#     robustifies the style component against domain shift, and fuses
#     them back with a learned spatial gate.
    
#     Args:
#         in_channels: Number of input feature channels (256 for Mask2Former)
#         kernel_size: Size of the learnable low-pass filter (default: 7)
#         init_sigma: Initial Gaussian sigma for the low-pass filter (default: 2.0)
#         reduction: Channel reduction ratio in the style robustifier (default: 4)
        
#     Input:  mask_features [B, C, H, W] from pixel decoder
#     Output: cleaned mask_features [B, C, H, W] for transformer decoder
    
#     Properties:
#         - Near-identity at initialization (safe for fine-tuning)
#         - Preserves channel magnitudes (einsum-compatible)
#         - Lightweight: ~0.1M additional parameters for C=256
#     """
    
#     def __init__(self, in_channels=256, kernel_size=7, init_sigma=2.0, reduction=4):
#         super().__init__()
        
#         self.in_channels = in_channels
        
#         # Step 1: Spatial frequency decomposition
#         self.low_pass = LearnableGaussianLowPass(
#             channels=in_channels,
#             kernel_size=kernel_size,
#             init_sigma=init_sigma,
#         )
#         # High-pass is computed as residual: style = x - content
        
#         # Step 2: Style robustification
#         self.style_robustifier = StyleRobustifier(
#             channels=in_channels,
#             reduction=reduction,
#         )
        
#         # Step 3: Gated fusion (spatial + channel aware)
#         self.gate = SpatioChannelGate(channels=in_channels)
    
#     def forward(self, mask_features):
#         """
#         Args:
#             mask_features: [B, C, H, W] from pixel decoder
#         Returns:
#             cleaned_features: [B, C, H, W] for transformer decoder
#         """
#         # Step 1: Decompose into content (low-freq) and style (high-freq)
#         content = self.low_pass(mask_features)
#         style = mask_features - content  # high-frequency residual
        
#         # Step 2: Robustify the style/texture component
#         style_robust = self.style_robustifier(style)
        
#         # Step 3: Gated fusion
#         gate = self.gate(content, style_robust)  # [B, 1, H, W]
        
#         # Recombine: content + gated style
#         cleaned = content + gate * style_robust
        
#         return cleaned
    
#     def extra_repr(self):
#         return (
#             f"in_channels={self.in_channels}, "
#             f"kernel_size={self.low_pass.kernel_size}, "
#             f"params={sum(p.numel() for p in self.parameters()) / 1e3:.1f}K"
#         )
 
 
# def build_csfd(cfg):
#     """Build CSFD module from detectron2 config."""
#     return CSFD(
#         in_channels=cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,  # 256 for Mask2Former
#         kernel_size=cfg.MODEL.CSFD.KERNEL_SIZE if hasattr(cfg.MODEL, 'CSFD') else 7,
#         init_sigma=cfg.MODEL.CSFD.INIT_SIGMA if hasattr(cfg.MODEL, 'CSFD') else 2.0,
#         reduction=cfg.MODEL.CSFD.REDUCTION if hasattr(cfg.MODEL, 'CSFD') else 4,
#     )
 
 
# # ============================================================================
# # Quick test / parameter count
# # ============================================================================
# if __name__ == "__main__":
#     # Test with Mask2Former's default mask_features shape
#     csfd = CSFD(in_channels=256)
    
#     # Count parameters
#     total_params = sum(p.numel() for p in csfd.parameters())
#     trainable_params = sum(p.numel() for p in csfd.parameters() if p.requires_grad)
#     print(f"CSFD Module:")
#     print(f"  Total parameters:     {total_params:,} ({total_params/1e3:.1f}K)")
#     print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e3:.1f}K)")
#     print(f"  Module structure:")
#     print(csfd)
    
#     # Test forward pass
#     x = torch.randn(2, 256, 128, 256)  # typical Cityscapes mask_features size
#     y = csfd(x)
#     print(f"\n  Input shape:  {x.shape}")
#     print(f"  Output shape: {y.shape}")
    
#     # Verify near-identity initialization
#     diff = (y - x).abs().mean().item()
#     print(f"  Mean absolute diff from input (should be small): {diff:.6f}")
    
#     # Verify channel magnitudes are preserved
#     in_chan_mag = x.mean(dim=[0, 2, 3])  # [C]
#     out_chan_mag = y.mean(dim=[0, 2, 3])
#     mag_corr = torch.corrcoef(torch.stack([in_chan_mag, out_chan_mag]))[0, 1].item()
#     print(f"  Channel magnitude correlation (should be ~1.0): {mag_corr:.6f}")
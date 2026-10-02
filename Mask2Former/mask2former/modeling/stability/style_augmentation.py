"""
style_augmentation.py
Style augmentation module for the stability regularization training.
 
Applies random colour · texture · lighting perturbations to a raw image
tensor to simulate domain shift while preserving semantic content.
 
Operates on raw (un-normalized) float32 image tensors in [0, 255] range,
CHW format, BGR channel order — exactly what Detectron2 supplies before
pixel_mean / pixel_std normalization in MaskFormer.preprocess_image().
 
Usage inside MaskFormer.forward():
    aug = StyleAugmentation(device=self.device)
    images_aug = aug(images_raw)           # same shape, [B, C, H, W]
    # then normalize images_aug normally
"""
 
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
class StyleAugmentation(nn.Module):
    """
    Differentiable-friendly style augmentation applied to a batch of raw
    image tensors.  All operations are in-place on [0, 255] float tensors.
 
    Augmentations (applied independently per image in the batch):
        1. Color jitter  — brightness, contrast, saturation, hue
        2. Random grayscale
        3. Gaussian blur  — smooths texture without changing semantics
 
    Design notes:
        - Probabilities and strengths are fixed at recommended defaults.
          Change via constructor args if ablating.
        - No flip / crop / resize — spatial content must be preserved so
          the decoder sees the same scene structure (just different style).
        - The module has no learnable parameters; it is a pure functional
          transform applied at training time only.
    """
 
    def __init__(
        self,
        # Color jitter
        brightness: float = 0.4,
        contrast:   float = 0.4,
        saturation: float = 0.4,
        hue:        float = 0.1,
        # Random grayscale
        grayscale_prob: float = 0.1,
        # Gaussian blur
        blur_prob:      float = 0.5,
        blur_sigma_min: float = 0.1,
        blur_sigma_max: float = 2.0,
        blur_kernel:    int   = 23,   # must be odd
    ):
        super().__init__()
        self.brightness   = brightness
        self.contrast     = contrast
        self.saturation   = saturation
        self.hue          = hue
        self.grayscale_prob = grayscale_prob
        self.blur_prob      = blur_prob
        self.blur_sigma_min = blur_sigma_min
        self.blur_sigma_max = blur_sigma_max
        self.blur_kernel    = blur_kernel
 
    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
 
    @staticmethod
    def _rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
        """img: [..., 3, H, W] RGB in [0, 1]. Returns HSV in [0, 1]."""
        r, g, b = img.unbind(dim=-3)
        maxc = torch.max(img, dim=-3).values
        minc = torch.min(img, dim=-3).values
        v    = maxc
        s    = torch.where(maxc > 0, (maxc - minc) / (maxc + 1e-8),
                           torch.zeros_like(maxc))
        rc   = (maxc - r) / (maxc - minc + 1e-8)
        gc   = (maxc - g) / (maxc - minc + 1e-8)
        bc   = (maxc - b) / (maxc - minc + 1e-8)
        h    = torch.where(r == maxc, bc - gc,
               torch.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
        h    = (h / 6.0) % 1.0
        h    = torch.where(minc == maxc, torch.zeros_like(h), h)
        return torch.stack([h, s, v], dim=-3)
 
    @staticmethod
    def _hsv_to_rgb(img: torch.Tensor) -> torch.Tensor:
        """img: [..., 3, H, W] HSV in [0, 1]. Returns RGB in [0, 1]."""
        h, s, v = img.unbind(dim=-3)
        h6 = h * 6.0
        i  = h6.long()
        f  = h6 - i.float()
        p  = v * (1 - s)
        q  = v * (1 - f * s)
        t  = v * (1 - (1 - f) * s)
        i  = i % 6
        rgb = torch.stack([
            torch.where(i == 0, v, torch.where(i == 1, q,
                torch.where(i == 2, p, torch.where(i == 3, p,
                    torch.where(i == 4, t, v))))),
            torch.where(i == 0, t, torch.where(i == 1, v,
                torch.where(i == 2, v, torch.where(i == 3, q,
                    torch.where(i == 4, p, p))))),
            torch.where(i == 0, p, torch.where(i == 1, p,
                torch.where(i == 2, t, torch.where(i == 3, v,
                    torch.where(i == 4, v, q))))),
        ], dim=-3)
        return rgb
 
    def _jitter_one(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: [3, H, W] float32 BGR in [0, 255].
        Returns jittered image in same format.
        """
        # Convert BGR [0,255] → RGB [0,1]
        rgb = img[[2, 1, 0], ...] / 255.0
 
        # Brightness
        factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        rgb = (rgb * factor).clamp(0, 1)
 
        # Contrast
        factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        mean   = rgb.mean(dim=[-2, -1], keepdim=True)
        rgb    = ((rgb - mean) * factor + mean).clamp(0, 1)
 
        # Saturation + hue (via HSV)
        hsv = self._rgb_to_hsv(rgb.unsqueeze(0)).squeeze(0)
        # Saturation
        s_factor = 1.0 + random.uniform(-self.saturation, self.saturation)
        hsv[1]   = (hsv[1] * s_factor).clamp(0, 1)
        # Hue shift
        h_shift  = random.uniform(-self.hue, self.hue)
        hsv[0]   = (hsv[0] + h_shift) % 1.0
        rgb = self._hsv_to_rgb(hsv.unsqueeze(0)).squeeze(0).clamp(0, 1)
 
        # Back to BGR [0, 255]
        bgr = rgb[[2, 1, 0], ...] * 255.0
        return bgr
 
    def _grayscale_one(self, img: torch.Tensor) -> torch.Tensor:
        """Convert to grayscale in BGR space (replicate across channels)."""
        # BGR weights: B=0.114, G=0.587, R=0.299 (same coefficients, BGR order)
        weights = img.new_tensor([0.114, 0.587, 0.299]).view(3, 1, 1)
        gray    = (img * weights).sum(dim=0, keepdim=True).expand_as(img)
        return gray
 
    def _gaussian_kernel(self, sigma: float, device) -> torch.Tensor:
        k   = self.blur_kernel
        half = k // 2
        xs  = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
        kernel_1d = torch.exp(-xs ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d
 
    def _blur_one(self, img: torch.Tensor) -> torch.Tensor:
        """img: [3, H, W]. Returns blurred image same shape."""
        sigma  = random.uniform(self.blur_sigma_min, self.blur_sigma_max)
        kernel = self._gaussian_kernel(sigma, img.device)  # [k, k]
        k      = self.blur_kernel
        # Apply as depthwise conv: [1, 3, H, W] with [3, 1, k, k] kernel
        kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3, 1, k, k)
        pad    = k // 2
        out    = F.conv2d(img.unsqueeze(0), kernel,
                          padding=pad, groups=3).squeeze(0)
        return out
 
    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------
 
    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] float32 BGR in [0, 255] (un-normalized).
 
        Returns:
            Augmented images, same shape and dtype, same device.
        """
        out = images.clone()
        B   = images.shape[0]
 
        for b in range(B):
            img = out[b]   # [3, H, W]
 
            # 1. Color jitter (always applied)
            img = self._jitter_one(img)
 
            # 2. Random grayscale
            if random.random() < self.grayscale_prob:
                img = self._grayscale_one(img)
 
            # 3. Gaussian blur
            if random.random() < self.blur_prob:
                img = self._blur_one(img)
 
            out[b] = img.clamp(0, 255)
 
        return out
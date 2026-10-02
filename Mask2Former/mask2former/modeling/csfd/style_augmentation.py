"""
Style Augmentation for CSFD Training
=====================================
 
Diverse photometric and atmospheric augmentations applied to training images
to make the backbone extract domain-invariant spatial features.
 
Design rationale (from diagnostic experiments):
- Night fails because backbone never saw dark scenes → add gamma/brightness reduction
- Rain fails because of local occlusions → add contrast disruption + noise
- Fog is mild but augmentation reinforces → add haze simulation
- GTA works because CSFD handles appearance → color jitter reinforces this
 
Applied at the IMAGE level during training, before backbone processing.
CSFD then cleans whatever residual domain artifacts remain at the decoder interface.
 
Usage:
    from mask2former.modeling.csfd.style_augmentation import StyleAugmentor
    
    augmentor = StyleAugmentor(severity=1.0)
    augmented_image = augmentor(image)  # numpy HWC uint8 → numpy HWC uint8
"""
 
import numpy as np
import random
from typing import Optional
 
 
class StyleAugmentor:
    """Applies random style augmentations to simulate domain shift.
    
    Each call randomly selects and applies a subset of augmentations.
    Augmentations are designed to cover the failure modes identified
    in diagnostic experiments (night, rain, fog, appearance shift).
    
    Args:
        severity: float in [0, 1], controls augmentation strength.
                  0 = no augmentation, 1 = full strength.
        prob: probability of applying any augmentation to a given image.
              Some images pass through unchanged for regularization.
        mode: "full" = all augmentations (original behavior)
              "targeted" = only augmentations that helped in experiments
                           (darkness, fog, gamma, noise — helped night/fog/snow)
                           Removes channel_shuffle and reduces contrast/saturation
                           which hurt BDD and rain.
    """
    
    def __init__(self, severity: float = 1.0, prob: float = 0.5, mode: str = "full"):
        self.severity = severity
        self.prob = prob
        self.mode = mode
        
        if mode == "full":
            # All augmentations — original behavior
            self.augmentations = [
                (self.random_brightness, 0.4),
                (self.random_contrast, 0.4),
                (self.random_saturation, 0.3),
                (self.random_gamma, 0.4),
                (self.random_hue_shift, 0.2),
                (self.gaussian_noise, 0.3),
                (self.gaussian_blur, 0.2),
                (self.simulate_fog, 0.15),
                (self.simulate_darkness, 0.15),
                (self.channel_shuffle, 0.1),
            ]
        elif mode == "targeted":
            # Only augmentations that helped night/fog/snow without hurting
            # BDD/rain. Derived from ablation: CSFD+StyleAug(full) gained
            # +3.96 night, +1.85 fog, +1.19 snow but lost -0.62 BDD, -2.74 rain.
            # Removing channel_shuffle and reducing contrast/saturation fixes this.
            self.augmentations = [
                (self.random_brightness, 0.3),     # mild, helps general robustness
                (self.random_contrast, 0.15),      # reduced — aggressive contrast hurt rain
                (self.random_gamma, 0.4),          # keep — critical for night simulation
                (self.gaussian_noise, 0.3),        # keep — sensor noise robustness
                (self.gaussian_blur, 0.15),        # mild — helps fog
                (self.simulate_fog, 0.2),          # keep — directly targets fog condition
                (self.simulate_darkness, 0.2),     # keep — directly targets night condition
                # REMOVED: channel_shuffle — disrupts color-dependent features, hurt BDD
                # REMOVED: random_hue_shift — similar issue to channel_shuffle
                # REDUCED: random_saturation — removed entirely, hurt rain reflections
            ]
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'full' or 'targeted'.")
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply random style augmentations.
        
        Args:
            image: numpy array, HWC, uint8, RGB format
        Returns:
            augmented image: same format
        """
        if random.random() > self.prob:
            return image
        
        # Convert to float for processing
        img = image.astype(np.float32)
        
        # Randomly apply a subset of augmentations
        for aug_fn, aug_prob in self.augmentations:
            if random.random() < aug_prob:
                img = aug_fn(img)
        
        # Clip and convert back
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    # ==================================================================
    # Photometric augmentations
    # ==================================================================
    
    def random_brightness(self, img: np.ndarray) -> np.ndarray:
        """Random brightness shift. Covers day/dusk/dawn variations."""
        s = self.severity
        factor = random.uniform(1.0 - 0.4 * s, 1.0 + 0.4 * s)
        return img * factor
    
    def random_contrast(self, img: np.ndarray) -> np.ndarray:
        """Random contrast change. Simulates camera exposure differences."""
        s = self.severity
        factor = random.uniform(1.0 - 0.4 * s, 1.0 + 0.4 * s)
        mean = img.mean()
        return (img - mean) * factor + mean
    
    def random_saturation(self, img: np.ndarray) -> np.ndarray:
        """Random saturation. Covers washed-out vs vivid scenes."""
        s = self.severity
        factor = random.uniform(1.0 - 0.6 * s, 1.0 + 0.4 * s)
        gray = img.mean(axis=2, keepdims=True)
        return gray + (img - gray) * factor
    
    def random_gamma(self, img: np.ndarray) -> np.ndarray:
        """Random gamma correction.
        
        Critical for night simulation: gamma > 1 darkens the image
        non-linearly, preserving some structure in bright regions
        while crushing dark regions — exactly like underexposure.
        """
        s = self.severity
        # gamma < 1 brightens, gamma > 1 darkens
        gamma = random.uniform(0.6, 1.0 + 1.0 * s)
        img_norm = img / 255.0
        img_gamma = np.power(img_norm + 1e-8, gamma) * 255.0
        return img_gamma
    
    def random_hue_shift(self, img: np.ndarray) -> np.ndarray:
        """Random hue rotation. Simulates different camera white balance."""
        s = self.severity
        # Simple channel-wise shift to approximate hue change
        shifts = np.array([
            random.uniform(-30 * s, 30 * s),
            random.uniform(-20 * s, 20 * s),
            random.uniform(-30 * s, 30 * s),
        ], dtype=np.float32)
        return img + shifts.reshape(1, 1, 3)
    
    # ==================================================================
    # Sensor noise simulation
    # ==================================================================
    
    def gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        """Additive Gaussian noise.
        
        Simulates high-ISO sensor noise in low-light conditions.
        Critical for night robustness — the backbone must learn to
        extract spatial structure through noise.
        """
        s = self.severity
        sigma = random.uniform(5, 25 * s)
        noise = np.random.randn(*img.shape).astype(np.float32) * sigma
        return img + noise
    
    def gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        """Gaussian blur. Simulates defocus, fog-like softening, rain on lens."""
        s = self.severity
        # Simple box blur approximation (no opencv dependency)
        kernel_size = random.choice([3, 5, 7])
        if kernel_size > 1:
            from scipy.ndimage import uniform_filter
            sigma = random.uniform(0.5, 2.0 * s)
            img = uniform_filter(img, size=(kernel_size, kernel_size, 1))
        return img
    
    # ==================================================================
    # Atmospheric simulation
    # ==================================================================
    
    def simulate_fog(self, img: np.ndarray) -> np.ndarray:
        """Simulate fog/haze using atmospheric scattering model.
        
        Based on Koschmieder's law: I(x) = J(x)·t(x) + A·(1 - t(x))
        where t(x) is transmission (how much original scene shows through),
        A is atmospheric light (bright sky color), J(x) is clean image.
        
        We use a simple uniform transmission for global fog effect.
        """
        s = self.severity
        # Atmospheric light (bright gray, slightly warm)
        A = np.array([220, 220, 210], dtype=np.float32).reshape(1, 1, 3)
        # Transmission: lower = thicker fog
        t = random.uniform(0.4, 1.0 - 0.3 * s)
        return img * t + A * (1.0 - t)
    
    def simulate_darkness(self, img: np.ndarray) -> np.ndarray:
        """Simulate nighttime / low-light conditions.
        
        Combines aggressive gamma darkening with localized brightness
        (simulating street lights) and elevated noise floor.
        This is the most critical augmentation for night robustness.
        """
        s = self.severity
        
        # Step 1: Aggressive gamma darkening
        gamma = random.uniform(1.5, 2.5 * s)
        img_norm = img / 255.0
        img_dark = np.power(img_norm + 1e-8, gamma) * 255.0
        
        # Step 2: Add noise floor (simulates high-ISO)
        noise_sigma = random.uniform(10, 30 * s)
        noise = np.random.randn(*img.shape).astype(np.float32) * noise_sigma
        img_dark = img_dark + noise
        
        # Step 3: Slight color cast (sodium vapor street light simulation)
        if random.random() < 0.5:
            # Warm orange cast
            cast = np.array([1.05, 0.95, 0.85], dtype=np.float32).reshape(1, 1, 3)
            img_dark = img_dark * cast
        
        return img_dark
    
    # ==================================================================
    # Color space augmentation
    # ==================================================================
    
    def channel_shuffle(self, img: np.ndarray) -> np.ndarray:
        """Random channel permutation.
        
        Forces the backbone to not rely on absolute color channel
        identity. A red car should be recognized whether R is in
        channel 0 or channel 2.
        """
        channels = list(range(3))
        random.shuffle(channels)
        return img[:, :, channels]
 
 
class ComposeStyleAugmentation:
    """Composes style augmentation with existing detectron2 transforms.
    
    This wraps the StyleAugmentor so it can be applied within the
    detectron2 data pipeline, before standard geometric augmentations.
    
    Usage in dataset mapper:
        style_aug = ComposeStyleAugmentation(severity=1.0, prob=0.5)
        # In __call__:
        image = style_aug(image)  # apply before other transforms
    """
    
    def __init__(self, severity: float = 1.0, prob: float = 0.5):
        self.augmentor = StyleAugmentor(severity=severity, prob=prob)
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.augmentor(image)
 
 
# ============================================================================
# Quick test
# ============================================================================
if __name__ == "__main__":
    # Create a dummy image
    img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
    
    aug = StyleAugmentor(severity=1.0, prob=1.0)
    
    print("StyleAugmentor test:")
    print(f"  Input:  shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")
    
    for i in range(5):
        out = aug(img)
        print(f"  Run {i}: shape={out.shape}, dtype={out.dtype}, range=[{out.min()}, {out.max()}]")
    
    print("OK")
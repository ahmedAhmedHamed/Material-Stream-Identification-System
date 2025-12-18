"""
augmentation.py

Data augmentation functions for image preprocessing.
Provides various augmentation techniques to increase dataset size and diversity.
Uses albumentations library for comprehensive augmentation pipeline.

Author: (your name)
"""
import cv2
import numpy as np
from typing import Iterable, Optional, Tuple
import albumentations as A
from albumentations.core.composition import Compose


# -----------------------------
# Legacy functions (backward compatibility)
# -----------------------------

def rotate_image(image: np.ndarray, angle_range: Tuple[float, float] = (-15, 15)) -> np.ndarray:
    """
    Rotate image by a random angle within the specified range.
    Legacy function for backward compatibility.
    
    Args:
        image: RGB uint8 image
        angle_range: Tuple of (min_angle, max_angle) in degrees
        
    Returns:
        Rotated RGB uint8 image
    """
    angle = np.random.uniform(angle_range[0], angle_range[1])
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                             flags=cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def flip_image(image: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
    """
    Flip image horizontally or vertically.
    Legacy function for backward compatibility.
    
    Args:
        image: RGB uint8 image
        direction: 'horizontal' or 'vertical'
        
    Returns:
        Flipped RGB uint8 image
    """
    if direction == 'horizontal':
        return cv2.flip(image, 1)
    elif direction == 'vertical':
        return cv2.flip(image, 0)
    else:
        raise ValueError(f"Invalid direction: {direction}. Use 'horizontal' or 'vertical'")


def adjust_brightness(image: np.ndarray, factor_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """
    Adjust image brightness by multiplying with a random factor.
    Legacy function for backward compatibility.
    
    Args:
        image: RGB uint8 image
        factor_range: Tuple of (min_factor, max_factor)
        
    Returns:
        Brightness-adjusted RGB uint8 image
    """
    factor = np.random.uniform(factor_range[0], factor_range[1])
    adjusted = image.astype(np.float32) * factor
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


def adjust_contrast(image: np.ndarray, factor_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """
    Adjust image contrast using cv2.convertScaleAbs.
    Legacy function for backward compatibility.
    
    Args:
        image: RGB uint8 image
        factor_range: Tuple of (min_factor, max_factor)
        
    Returns:
        Contrast-adjusted RGB uint8 image
    """
    factor = np.random.uniform(factor_range[0], factor_range[1])
    adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return adjusted


def add_noise(image: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """
    Add Gaussian noise to the image.
    Legacy function for backward compatibility.
    
    Args:
        image: RGB uint8 image
        noise_level: Standard deviation of noise (relative to 255)
        
    Returns:
        Noisy RGB uint8 image
    """
    noise_std = noise_level * 255
    noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


# -----------------------------
# Albumentations pipeline
# -----------------------------

def create_geometric_augmentations() -> list:
    """
    Create geometric augmentation transforms.
    
    Returns:
        List of geometric augmentation transforms
    """
    return [
        A.Rotate(limit=15, p=0.6, border_mode=cv2.BORDER_REPLICATE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Perspective(scale=(0.05, 0.1), p=0.3, pad_mode=cv2.BORDER_REPLICATE),
        A.ElasticTransform(alpha=50, sigma=5, p=0.2, border_mode=cv2.BORDER_REPLICATE),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2, border_mode=cv2.BORDER_REPLICATE),
    ]


def create_color_augmentations() -> list:
    """
    Create color augmentation transforms.
    
    Returns:
        List of color augmentation transforms
    """
    return [
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.6
        ),
        A.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.5
        ),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    ]


def create_blur_noise_augmentations() -> list:
    """
    Create blur and noise augmentation transforms.
    
    Returns:
        List of blur/noise augmentation transforms
    """
    return [
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.2),
    ]


def create_advanced_augmentations() -> list:
    """
    Create advanced augmentation transforms.
    
    Returns:
        List of advanced augmentation transforms
    """
    return [
        A.CoarseDropout(
            max_holes=8, max_height=16, max_width=16, p=0.2
        ),
        A.OpticalDistortion(
            distort_limit=0.1, shift_limit=0.05, p=0.2, border_mode=cv2.BORDER_REPLICATE
        ),
    ]


# -----------------------------
# Named augmentation registry (moderate intensity)
# -----------------------------

MODERATE_AUGMENTATIONS: dict[str, A.BasicTransform] = {
    # Geometric
    "rotate_15": A.Rotate(limit=15, p=1.0, border_mode=cv2.BORDER_REPLICATE),
    "hflip": A.HorizontalFlip(p=1.0),
    "vflip": A.VerticalFlip(p=1.0),
    "perspective": A.Perspective(scale=(0.05, 0.1), p=1.0, pad_mode=cv2.BORDER_REPLICATE),
    "elastic": A.ElasticTransform(alpha=50, sigma=5, p=1.0, border_mode=cv2.BORDER_REPLICATE),
    "grid_distortion": A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0, border_mode=cv2.BORDER_REPLICATE),
    # Color
    "brightness_contrast": A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    "color_jitter": A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=1.0),
    "hsv": A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=1.0),
    "rgb_shift": A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
    "clahe": A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    # Blur / noise
    "gaussian_blur": A.GaussianBlur(blur_limit=(3, 5), p=1.0),
    "motion_blur": A.MotionBlur(blur_limit=5, p=1.0),
    "gauss_noise": A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
    "iso_noise": A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
    # Advanced
    "coarse_dropout": A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=1.0),
    "optical_distortion": A.OpticalDistortion(
        distort_limit=0.1, shift_limit=0.05, p=1.0, border_mode=cv2.BORDER_REPLICATE
    ),
}


def available_moderate_augmentations() -> tuple[str, ...]:
    return tuple(sorted(MODERATE_AUGMENTATIONS.keys()))


def create_augmentation_pipeline(
    intensity: str = "moderate",
    *,
    allowed_groups: Iterable[str] | None = None,
    allowed_augmentations: Iterable[str] | None = None,
) -> Compose:
    """
    Create augmentation pipeline using albumentations.
    Applies only one augmentation at a time (randomly selected).
    
    Args:
        intensity: Augmentation intensity level. Options:
                  'light': Fewer, milder augmentations
                  'moderate': Balanced augmentations (default)
                  'strong': More aggressive augmentations
        allowed_groups: Optional subset of augmentation groups to use when intensity='moderate'.
                        Supported values: 'geometric', 'color', 'blur_noise', 'advanced'.
        allowed_augmentations: Optional allow-list of specific augmentation keys (moderate only).
                               If provided, it overrides allowed_groups. See available_moderate_augmentations().
    
    Returns:
        Albumentations Compose pipeline that applies one augmentation
    """
    if intensity == 'light':
        # Light augmentations - fewer and milder
        all_augmentations = [
            A.Rotate(limit=10, p=1.0),
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=1.0
            ),
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
        ]
    elif intensity == 'moderate':
        # Moderate augmentations - balanced for material recognition
        keys = _normalize_allowed_augmentations(allowed_augmentations)
        if keys is not None:
            all_augmentations = [MODERATE_AUGMENTATIONS[k] for k in keys]
        else:
            groups = _normalize_allowed_groups(allowed_groups)
            all_augmentations = _build_moderate_augmentations(groups)
        # Set p=1.0 for all transforms since OneOf will handle selection
        for aug in all_augmentations:
            aug.p = 1.0
    elif intensity == 'strong':
        # Strong augmentations - more aggressive
        all_augmentations = [
            A.Rotate(limit=25, p=1.0, border_mode=cv2.BORDER_REPLICATE),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Perspective(scale=(0.1, 0.15), p=1.0, pad_mode=cv2.BORDER_REPLICATE),
            A.ElasticTransform(alpha=80, sigma=8, p=1.0, border_mode=cv2.BORDER_REPLICATE),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=1.0
            ),
            A.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1, p=1.0
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(var_limit=(15.0, 50.0), p=1.0),
            A.CoarseDropout(
                max_holes=12, max_height=24, max_width=24, p=1.0
            ),
        ]
    else:
        raise ValueError(
            f"Invalid intensity: {intensity}. Use 'light', 'moderate', or 'strong'"
        )
    
    # Use OneOf to ensure only one augmentation is applied at a time
    return A.Compose([A.OneOf(all_augmentations, p=1.0)], p=1.0)


# Global pipeline instances (lazy initialization)
_augmentation_pipelines: dict[tuple[str, tuple[str, ...], tuple[str, ...]], Compose] = {}


def get_augmentation_pipeline(
    intensity: str = "moderate",
    *,
    allowed_groups: Iterable[str] | None = None,
    allowed_augmentations: Iterable[str] | None = None,
) -> Compose:
    """
    Get or create augmentation pipeline (cached by intensity).
    
    Args:
        intensity: Augmentation intensity level
        allowed_groups: Optional subset of augmentation groups (see create_augmentation_pipeline)
        allowed_augmentations: Optional allow-list of specific augmentation keys (see create_augmentation_pipeline)
        
    Returns:
        Augmentation pipeline
    """
    global _augmentation_pipelines
    groups = tuple(_normalize_allowed_groups(allowed_groups))
    keys = tuple(_normalize_allowed_augmentations(allowed_augmentations) or ())
    key = (intensity, groups, keys)
    if key not in _augmentation_pipelines:
        _augmentation_pipelines[key] = create_augmentation_pipeline(
            intensity,
            allowed_groups=groups,
            allowed_augmentations=keys or None,
        )
    return _augmentation_pipelines[key]


def augment_image(
    image: np.ndarray,
    technique: str = "random",
    intensity: str = "moderate",
    *,
    allowed_groups: Iterable[str] | None = None,
    allowed_augmentations: Iterable[str] | None = None,
) -> np.ndarray:
    """
    Apply a single augmentation to the image using albumentations pipeline.
    Only one augmentation is applied at a time (randomly selected).
    Maintains backward compatibility with legacy 'technique' parameter.
    
    Args:
        image: RGB uint8 image (H, W, C)
        technique: Legacy parameter for backward compatibility.
                  'random' uses albumentations pipeline (recommended).
                  Other values fall back to legacy single-augmentation mode.
        intensity: Augmentation intensity ('light', 'moderate', 'strong').
                  Only used when technique='random'.
        allowed_groups: Optional subset of augmentation groups to use when technique='random'
                        and intensity='moderate'.
        allowed_augmentations: Optional allow-list of specific augmentation keys (moderate only).
                  
    Returns:
        Augmented RGB uint8 image
    """
    if technique == 'random':
        # Use comprehensive albumentations pipeline
        pipeline = get_augmentation_pipeline(
            intensity,
            allowed_groups=allowed_groups,
            allowed_augmentations=allowed_augmentations,
        )
        augmented = pipeline(image=image)['image']
        return augmented
    else:
        # Fall back to legacy single-augmentation mode for backward compatibility
        if technique == 'rotate':
            return rotate_image(image)
        elif technique == 'flip':
            direction = np.random.choice(['horizontal', 'vertical'])
            return flip_image(image, direction)
        elif technique == 'brightness':
            return adjust_brightness(image)
        elif technique == 'contrast':
            return adjust_contrast(image)
        elif technique == 'noise':
            return add_noise(image)
        else:
            raise ValueError(f"Unknown technique: {technique}")


def _normalize_allowed_groups(allowed_groups: Iterable[str] | None) -> list[str]:
    if allowed_groups is None:
        return ["advanced", "blur_noise", "color", "geometric"]
    groups = sorted({str(g).strip() for g in allowed_groups if str(g).strip()})
    supported = {"geometric", "color", "blur_noise", "advanced"}
    unknown = [g for g in groups if g not in supported]
    if unknown:
        raise ValueError(f"Unknown augmentation groups: {unknown}. Supported: {sorted(supported)}")
    return groups


def _normalize_allowed_augmentations(allowed_augmentations: Iterable[str] | None) -> list[str] | None:
    if allowed_augmentations is None:
        return None
    keys = sorted({str(k).strip() for k in allowed_augmentations if str(k).strip()})
    unknown = [k for k in keys if k not in MODERATE_AUGMENTATIONS]
    if unknown:
        raise ValueError(
            f"Unknown augmentation keys: {unknown}. Available: {list(available_moderate_augmentations())}"
        )
    if not keys:
        raise ValueError("No augmentations selected (allowed_augmentations was empty).")
    return keys


def _build_moderate_augmentations(groups: Iterable[str]) -> list:
    aug: list = []
    for g in groups:
        if g == "geometric":
            aug += create_geometric_augmentations()
        elif g == "color":
            aug += create_color_augmentations()
        elif g == "blur_noise":
            aug += create_blur_noise_augmentations()
        elif g == "advanced":
            aug += create_advanced_augmentations()
    if not aug:
        raise ValueError("No augmentations selected (allowed_groups was empty).")
    return aug

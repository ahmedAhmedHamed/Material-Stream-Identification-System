"""
augmentation.py

Data augmentation functions for image preprocessing.
Provides various augmentation techniques to increase dataset size and diversity.

Author: (your name)
"""
import cv2
import numpy as np
from typing import Tuple


def rotate_image(image: np.ndarray, angle_range: Tuple[float, float] = (-15, 15)) -> np.ndarray:
    """
    Rotate image by a random angle within the specified range.
    
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


def augment_image(image: np.ndarray, technique: str = 'random') -> np.ndarray:
    """
    Apply a single augmentation technique to the image.
    
    Args:
        image: RGB uint8 image
        technique: Augmentation technique to apply. Options:
                  'random', 'rotate', 'flip', 'brightness', 'contrast', 'noise'
                  
    Returns:
        Augmented RGB uint8 image
    """
    if technique == 'random':
        techniques = ['rotate', 'flip', 'brightness', 'contrast', 'noise']
        technique = np.random.choice(techniques)
    
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

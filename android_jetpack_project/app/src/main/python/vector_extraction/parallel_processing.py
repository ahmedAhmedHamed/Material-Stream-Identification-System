"""
parallel_processing.py

Parallel processing utilities for feature extraction.
Provides multithreaded processing of images to speed up feature extraction.

Author: (your name)
"""
import random
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np


# -----------------------------
# Parallel processing helpers
# -----------------------------

def process_images_parallel(image_paths: List[str]) -> List[Tuple[np.ndarray, str]]:
    """
    Process multiple images in parallel.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        List of tuples (feature_vector, label)
    """
    from vector_extraction.feature_extractor import process_image_for_features, MAX_WORKERS
    
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {
            executor.submit(process_image_for_features, path): path 
            for path in image_paths
        }
        
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Image {path} generated an exception: {exc}")
                raise
    
    return results


def process_augmented_image(args: Tuple[np.ndarray, str]) -> Tuple[np.ndarray, str]:
    """
    Extract features from an augmented image.
    Helper function for parallel processing.
    
    Args:
        args: Tuple of (augmented_image, label)
        
    Returns:
        Tuple of (feature_vector, label)
    """
    from vector_extraction.feature_extractor import extract_features
    
    aug_image, label = args
    features = extract_features(aug_image)
    return features, label


def generate_augmented_images(image_paths: List[str], needed_count: int) -> List[Tuple[np.ndarray, str]]:
    """
    Generate augmented image samples (without feature extraction).
    
    Args:
        image_paths: List of image file paths for a single class
        needed_count: Number of augmented samples to generate
        
    Returns:
        List of tuples (augmented_image, label) where image is RGB uint8
    """
    from vector_extraction.feature_extractor import get_label_from_path
    from vector_extraction.augmentation import augment_image
    
    augmented_samples = []
    label = get_label_from_path(image_paths[0])
    
    for _ in range(needed_count):
        source_path = random.choice(image_paths)
        image = cv2.imread(source_path)
        if image is None:
            raise IOError(f"Could not read image: {source_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = augment_image(image, technique='random')
        augmented_samples.append((augmented, label))
    
    return augmented_samples


def extract_features_from_augmented_parallel(augmented_samples: List[Tuple[np.ndarray, str]]) -> List[Tuple[np.ndarray, str]]:
    """
    Extract features from augmented images in parallel.
    
    Args:
        augmented_samples: List of tuples (augmented_image, label)
        
    Returns:
        List of tuples (feature_vector, label)
    """
    from vector_extraction.feature_extractor import MAX_WORKERS
    
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_sample = {
            executor.submit(process_augmented_image, sample): sample 
            for sample in augmented_samples
        }
        
        for future in as_completed(future_to_sample):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Augmented image processing generated an exception: {exc}")
                raise
    
    return results


def generate_augmented_samples_parallel(image_paths: List[str], 
                                        needed_count: int) -> List[Tuple[np.ndarray, str]]:
    """
    Generate augmented image samples and extract features in parallel.
    
    Args:
        image_paths: List of image file paths for a single class
        needed_count: Number of augmented samples to generate
        
    Returns:
        List of tuples (feature_vector, label)
    """
    augmented_samples = generate_augmented_images(image_paths, needed_count)
    return extract_features_from_augmented_parallel(augmented_samples)


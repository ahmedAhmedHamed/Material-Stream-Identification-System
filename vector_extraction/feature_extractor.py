"""
classical_feature_extraction.py

Classical feature extraction pipeline for material recognition.
Outputs fixed-size feature vectors suitable for SVM and k-NN.

Features:
- HSV color histogram
- Local Binary Patterns (LBP)
- Gabor texture filters
- Edge density

Author: (your name)
"""
import os
from pathlib import Path
from typing import List, Dict, Tuple
import random

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from vector_extraction.augmentation import augment_image
from vector_extraction.parallel_processing import (
    process_images_parallel,
    generate_augmented_samples_parallel
)


# -----------------------------
# Configuration
# -----------------------------

IMAGE_SIZE = (256, 256)

HSV_BINS = (8, 8, 8)       # 512 dims
LBP_POINTS = 8
LBP_RADIUS = 1             # 59 dims (uniform)
GABOR_ORIENTATIONS = 4     # 8 dims (mean + std per orientation)

TOTAL_FEATURE_DIM = 512 + 59 + (2 * GABOR_ORIENTATIONS) + 1

MAX_WORKERS = 30  # Number of threads for parallel processing


# -----------------------------
# Preprocessing
# -----------------------------

def preprocess_image(image):
    """
    Resize image to a fixed size.
    Input: RGB uint8 image
    Output: RGB uint8 image
    """
    return cv2.resize(image, IMAGE_SIZE)


# -----------------------------
# Feature extractors
# -----------------------------

def extract_hsv_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2],
        None,
        HSV_BINS,
        [0, 180, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(
        gray,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method="uniform"
    )
    hist, _ = np.histogram(lbp, bins=59, range=(0, 59))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist


def extract_gabor(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = []

    for theta in np.linspace(0, np.pi, GABOR_ORIENTATIONS, endpoint=False):
        kernel = cv2.getGaborKernel(
            ksize=(31, 31),
            sigma=4.0,
            theta=theta,
            lambd=10.0,
            gamma=0.5,
            psi=0
        )
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        features.append(filtered.mean())
        features.append(filtered.std())

    return np.array(features, dtype=np.float32)


def extract_edge_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.array([edges.mean()], dtype=np.float32)


# -----------------------------
# Full feature vector
# -----------------------------

def extract_features(image):
    """
    Input: RGB uint8 image
    Output: 1D NumPy array of fixed length
    """
    image = preprocess_image(image)

    features = np.concatenate([
        extract_hsv_histogram(image),
        extract_lbp(image),
        extract_gabor(image),
        # extract_edge_density(image)
    ])

    # assert features.shape[0] == TOTAL_FEATURE_DIM
    return features


# -----------------------------
# Dataset-level extraction
# -----------------------------

def get_label_from_path(image_path):
    return Path(image_path).parent.name


def analyze_class_distribution(image_paths: List[str]) -> Dict[str, List[str]]:
    """
    Group image paths by their class label.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        Dictionary mapping class labels to lists of image paths
    """
    class_to_paths = {}
    for path in image_paths:
        label = get_label_from_path(path)
        if label not in class_to_paths:
            class_to_paths[label] = []
        class_to_paths[label].append(path)
    return class_to_paths


def generate_augmented_samples(image_paths: List[str], needed_count: int) -> List[Tuple[np.ndarray, str]]:
    """
    Generate augmented image samples by randomly selecting from existing images
    and applying random augmentations.
    
    Args:
        image_paths: List of image file paths for a single class
        needed_count: Number of augmented samples to generate
        
    Returns:
        List of tuples (augmented_image, label) where image is RGB uint8
    """
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


def process_image_for_features(image_path: str) -> Tuple[np.ndarray, str]:
    """
    Load image and extract features.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (feature_vector, label)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not read image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    features = extract_features(image)
    label = get_label_from_path(image_path)
    
    return features, label


def process_validation_set(class_to_val_paths: Dict[str, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process validation images without augmentation (parallelized).
    
    Args:
        class_to_val_paths: Dictionary mapping class labels to validation image paths
        
    Returns:
        Tuple of (X_val, y_val) arrays
    """
    all_paths = []
    for paths in class_to_val_paths.values():
        all_paths.extend(paths)
    
    results = process_images_parallel(all_paths)
    
    X_val = [features for features, _ in results]
    y_val = [label for _, label in results]
    
    return np.array(X_val), np.array(y_val)


def process_training_set(class_to_train_paths: Dict[str, List[str]], 
                        augmentation_multiplier: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process training images (original + uniformly augmented) with parallelization.
    Applies the same augmentation multiplier to all classes, preserving original class distribution.
    
    Args:
        class_to_train_paths: Dictionary mapping class labels to training image paths
        augmentation_multiplier: Multiplier for augmentation (default: 1.0 = no augmentation)
                                 e.g., 1.0 = no augmentation, 2.0 = double each class
        
    Returns:
        Tuple of (X_train, y_train) arrays
    """
    X_train = []
    y_train = []
    
    for label, paths in class_to_train_paths.items():
        original_count = len(paths)
        augmented_count = int(original_count * augmentation_multiplier)
        
        # Process original images in parallel
        original_results = process_images_parallel(paths)
        for features, label_val in original_results:
            X_train.append(features)
            y_train.append(label_val)
        
        # Process augmented images in parallel
        if augmented_count > 0:
            augmented_results = generate_augmented_samples_parallel(
                paths, augmented_count
            )
            for features, label_val in augmented_results:
                X_train.append(features)
                y_train.append(label_val)
    
    return np.array(X_train), np.array(y_train)


def build_feature_matrix(image_paths: List[str], 
                         train_output_path: str = "../features_train.npz",
                         val_output_path: str = "../features_val.npz",
                         test_size: float = 0.3,
                         random_state: int = 42,
                         augmentation_multiplier: float = 1.0):
    """
    Build feature matrix with uniform data augmentation and split into train/validation.
    Validation dataset is 30% of original data (pre-augmentation) and is not augmented.
    Training dataset is 70% of original data plus uniformly augmented samples.
    All classes are augmented by the same multiplier, preserving original class distribution.
    Now with parallel processing support.
    
    Args:
        image_paths: list of image file paths
        train_output_path: path to save training .npz file
        val_output_path: path to save validation .npz file
        test_size: proportion of original data for validation (default: 0.3)
        random_state: random seed for reproducibility (default: 42)
        augmentation_multiplier: Multiplier for uniform augmentation (default: 1.0 = no augmentation)
                                  e.g., 1.0 = no augmentation, 2.0 = double each class
        
    Returns:
        tuple of (X_train, y_train, X_val, y_val) arrays
    """
    class_to_paths = analyze_class_distribution(image_paths)
    
    class_to_train_paths = {}
    class_to_val_paths = {}
    
    for label, paths in class_to_paths.items():
        train_paths, val_paths = train_test_split(
            paths, test_size=test_size, random_state=random_state
        )
        class_to_train_paths[label] = train_paths
        class_to_val_paths[label] = val_paths
    
    X_val, y_val = process_validation_set(class_to_val_paths)
    X_train, y_train = process_training_set(
        class_to_train_paths, 
        augmentation_multiplier=augmentation_multiplier
    )
    
    np.savez(train_output_path, X=X_train, y=y_train)
    np.savez(val_output_path, X=X_val, y=y_val)
    
    return X_train, y_train, X_val, y_val


# -----------------------------
# Feature scaling
# -----------------------------

def scale_features(X_train, X_val=None):
    """
    Standardize features (required for SVM / k-NN).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_val_scaled, scaler

    return X_train_scaled, scaler

def get_image_paths(root):
    exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    return [str(p) for p in Path(root).rglob("*") if p.suffix.lower() in exts]


def print_class_distribution(class_to_paths: Dict[str, List[str]], title: str = "Class Distribution"):
    """
    Print class distribution statistics.
    
    Args:
        class_to_paths: Dictionary mapping class labels to image paths
        title: Title for the distribution report
    """
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    
    class_sizes = {label: len(paths) for label, paths in class_to_paths.items()}
    sorted_classes = sorted(class_sizes.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTotal classes: {len(class_sizes)}")
    print(f"Total images: {sum(class_sizes.values())}")
    print(f"\nClass sizes:")
    for label, size in sorted_classes:
        print(f"  {label:15s}: {size:4d} images")
    
    sizes = list(class_sizes.values())
    print(f"\nStatistics:")
    print(f"  Min:  {min(sizes)}")
    print(f"  Max:  {max(sizes)}")
    print(f"  Mean: {int(np.mean(sizes)):.1f}")
    print(f"  Median: {int(np.median(sizes))}")
    print(f"  Imbalance ratio (max/min): {max(sizes) / min(sizes):.2f}x")


# -----------------------------
# Example usage
# -----------------------------

if __name__ == "__main__":
    print(f"Classical feature extractor ready.")
    print(f"Feature dimensionality: {TOTAL_FEATURE_DIM}")
    
    image_paths = get_image_paths('../dataset')
    class_to_paths = analyze_class_distribution(image_paths)
    print_class_distribution(class_to_paths, "Original Dataset Distribution")
    
    # Apply uniform augmentation: all classes augmented by the same multiplier
    # e.g., 1.0 = no augmentation, 2.0 = double each class (preserves imbalance)
    augmentation_multiplier = 1.0  # Change to 2.0 to double all classes, etc.
    X_train, y_train, X_val, y_val = build_feature_matrix(
        image_paths,
        augmentation_multiplier=augmentation_multiplier
    )
    
    print(f"\n{'='*60}")
    print("Feature Extraction Complete")
    print(f"{'='*60}")
    print(f"Augmentation multiplier: {augmentation_multiplier}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Show final training distribution
    from collections import Counter
    train_dist = Counter(y_train)
    print(f"\nFinal training distribution:")
    for label, count in sorted(train_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label:15s}: {count:4d} samples")

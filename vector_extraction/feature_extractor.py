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

from vector_extraction.augmentation import augment_image


# -----------------------------
# Configuration
# -----------------------------

IMAGE_SIZE = (256, 256)

HSV_BINS = (8, 8, 8)       # 512 dims
LBP_POINTS = 8
LBP_RADIUS = 1             # 59 dims (uniform)
GABOR_ORIENTATIONS = 4     # 8 dims (mean + std per orientation)

TOTAL_FEATURE_DIM = 512 + 59 + (2 * GABOR_ORIENTATIONS) + 1


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
        extract_edge_density(image)
    ])

    assert features.shape[0] == TOTAL_FEATURE_DIM
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


def build_feature_matrix(image_paths: List[str], output_path: str = "../features.npz"):
    """
    Build feature matrix with data augmentation.
    Doubles dataset size and balances all classes to equal size (2x largest class).
    
    Args:
        image_paths: list of image file paths
        output_path: path to save the .npz file containing X and y arrays
        
    Returns:
        tuple of (X, y) arrays
    """
    class_to_paths = analyze_class_distribution(image_paths)
    max_class_size = max(len(paths) for paths in class_to_paths.values())
    target_size = max_class_size * 2
    
    X = []
    y = []
    
    for label, paths in class_to_paths.items():
        current_count = len(paths)
        needed_count = target_size - current_count
        
        for path in paths:
            features, label_val = process_image_for_features(path)
            X.append(features)
            y.append(label_val)
        
        if needed_count > 0:
            augmented_samples = generate_augmented_samples(paths, needed_count)
            for aug_image, label_val in augmented_samples:
                features = extract_features(aug_image)
                X.append(features)
                y.append(label_val)
    
    X = np.array(X)
    y = np.array(y)
    
    np.savez(output_path, X=X, y=y)
    
    return X, y


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


# -----------------------------
# Example usage
# -----------------------------

if __name__ == "__main__":
    print(f"Classical feature extractor ready.")
    print(f"Feature dimensionality: {TOTAL_FEATURE_DIM}")
    image_paths = get_image_paths('../dataset')
    X, Y = build_feature_matrix(image_paths)

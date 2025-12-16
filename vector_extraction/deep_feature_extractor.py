"""
deep_feature_extractor.py

Deep learning feature extraction pipeline for material recognition.
Uses pre-trained ResNet50 to extract fixed-size feature vectors suitable for SVM and k-NN.

Features:
- ResNet50 pre-trained on ImageNet
- 2048-dimensional feature vectors from penultimate layer

Author: (your name)
"""
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
import threading

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from vector_extraction.augmentation import augment_image


# -----------------------------
# Configuration
# -----------------------------

IMAGE_SIZE = (224, 224)  # ResNet50 standard input size
FEATURE_DIM = 2048  # ResNet50 penultimate layer dimension
BATCH_SIZE = 32  # Batch size for efficient processing
MAX_WORKERS = 10  # Reduced for deep model (GPU memory considerations)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# -----------------------------
# Model initialization (thread-safe)
# -----------------------------

_model_lock = threading.Lock()
_model = None


def get_model():
    """
    Get or initialize ResNet50 model in thread-safe manner.
    Model is loaded once and reused.
    
    Returns:
        ResNet50 model with classification layer removed
    """
    global _model
    
    with _model_lock:
        if _model is None:
            # Load pre-trained ResNet50
            model = models.resnet50(weights='IMAGENET1K_V2')
            
            # Remove the final classification layer (fc)
            # Keep everything up to avgpool (which outputs 2048 features)
            # ResNet50 structure: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
            model = nn.Sequential(*list(model.children())[:-1])
            
            # Set to evaluation mode
            model.eval()
            
            # Move to device
            model = model.to(DEVICE)
            
            _model = model
    
    return _model


# -----------------------------
# Preprocessing
# -----------------------------

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for ResNet50: resize and normalize.
    
    Args:
        image: RGB uint8 image (H, W, C)
        
    Returns:
        Preprocessed image as numpy array (C, H, W) normalized for ImageNet
    """
    # Resize to 224x224
    resized = cv2.resize(image, IMAGE_SIZE)
    
    # Convert to float32 and normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD).reshape(1, 1, 3)
    normalized = (normalized - mean) / std
    
    # Convert to CHW format (C, H, W)
    normalized = np.transpose(normalized, (2, 0, 1))
    
    return normalized


def preprocess_batch(images: List[np.ndarray]) -> torch.Tensor:
    """
    Preprocess a batch of images for ResNet50.
    
    Args:
        images: List of RGB uint8 images
        
    Returns:
        Torch tensor of shape (batch_size, 3, 224, 224)
    """
    preprocessed = [preprocess_image(img) for img in images]
    batch_tensor = torch.from_numpy(np.stack(preprocessed)).float()
    return batch_tensor


# -----------------------------
# Feature extraction
# -----------------------------

def extract_features_single(image: np.ndarray) -> np.ndarray:
    """
    Extract features from a single image using ResNet50.
    
    Args:
        image: RGB uint8 image (H, W, C)
        
    Returns:
        1D NumPy array of 2048 features
    """
    model = get_model()
    
    # Preprocess image
    preprocessed = preprocess_image(image)
    
    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(preprocessed).float().unsqueeze(0)
    tensor = tensor.to(DEVICE)
    
    # Extract features
    with torch.no_grad():
        features = model(tensor)
        # Remove batch dimension and flatten
        features = features.squeeze().cpu().numpy().flatten()
    
    return features.astype(np.float32)


def extract_features_batch(images: List[np.ndarray]) -> np.ndarray:
    """
    Extract features from a batch of images using ResNet50.
    
    Args:
        images: List of RGB uint8 images
        
    Returns:
        2D NumPy array of shape (batch_size, 2048)
    """
    model = get_model()
    
    # Preprocess batch
    batch_tensor = preprocess_batch(images)
    batch_tensor = batch_tensor.to(DEVICE)
    
    # Extract features
    with torch.no_grad():
        features = model(batch_tensor)
        # Reshape to (batch_size, 2048)
        features = features.view(features.size(0), -1)
        features = features.cpu().numpy()
    
    return features.astype(np.float32)


def extract_features(image: np.ndarray) -> np.ndarray:
    """
    Extract features from an image (wrapper for compatibility).
    
    Args:
        image: RGB uint8 image
        
    Returns:
        1D NumPy array of fixed length (2048)
    """
    return extract_features_single(image)


# -----------------------------
# Dataset-level extraction
# -----------------------------

def get_label_from_path(image_path: str) -> str:
    """Extract class label from image path."""
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


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from path and convert to RGB.
    
    Args:
        image_path: Path to image file
        
    Returns:
        RGB uint8 image
    """
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def process_image_for_features(image_path: str) -> Tuple[np.ndarray, str]:
    """
    Load image and extract features.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (feature_vector, label)
    """
    image = load_image(image_path)
    features = extract_features(image)
    label = get_label_from_path(image_path)
    return features, label


def process_images_batch(image_paths: List[str]) -> List[Tuple[np.ndarray, str]]:
    """
    Process a batch of images efficiently.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        List of tuples (feature_vector, label)
    """
    # Load all images
    images = []
    labels = []
    for path in image_paths:
        try:
            image = load_image(path)
            images.append(image)
            labels.append(get_label_from_path(path))
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
    
    if not images:
        return []
    
    # Extract features in batch
    features = extract_features_batch(images)
    
    # Combine with labels
    results = [(features[i], labels[i]) for i in range(len(images))]
    return results


def process_images_parallel(image_paths: List[str]) -> List[Tuple[np.ndarray, str]]:
    """
    Process multiple images in parallel using batch processing.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        List of tuples (feature_vector, label)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = []
    
    # Split into batches
    batches = [image_paths[i:i + BATCH_SIZE] 
               for i in range(0, len(image_paths), BATCH_SIZE)]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {
            executor.submit(process_images_batch, batch): batch 
            for batch in batches
        }
        
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as exc:
                print(f"Batch processing generated an exception: {exc}")
                raise
    
    return results


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
        image = load_image(source_path)
        augmented = augment_image(image, technique='random')
        augmented_samples.append((augmented, label))
    
    return augmented_samples


def process_augmented_image(args: Tuple[np.ndarray, str]) -> Tuple[np.ndarray, str]:
    """
    Extract features from an augmented image.
    Helper function for parallel processing.
    
    Args:
        args: Tuple of (augmented_image, label)
        
    Returns:
        Tuple of (feature_vector, label)
    """
    aug_image, label = args
    features = extract_features(aug_image)
    return features, label


def process_augmented_batch(augmented_samples: List[Tuple[np.ndarray, str]]) -> List[Tuple[np.ndarray, str]]:
    """
    Process a batch of augmented images.
    
    Args:
        augmented_samples: List of tuples (augmented_image, label)
        
    Returns:
        List of tuples (feature_vector, label)
    """
    images = [img for img, _ in augmented_samples]
    labels = [label for _, label in augmented_samples]
    
    features = extract_features_batch(images)
    
    results = [(features[i], labels[i]) for i in range(len(images))]
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
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Generate augmented images
    augmented_samples = generate_augmented_samples(image_paths, needed_count)
    
    # Process in batches
    results = []
    batches = [augmented_samples[i:i + BATCH_SIZE] 
               for i in range(0, len(augmented_samples), BATCH_SIZE)]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {
            executor.submit(process_augmented_batch, batch): batch 
            for batch in batches
        }
        
        for future in as_completed(future_to_batch):
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as exc:
                print(f"Augmented batch processing generated an exception: {exc}")
                raise
    
    return results


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
    """Get all image paths from directory tree."""
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
    print(f"Deep feature extractor ready.")
    print(f"Using device: {DEVICE}")
    print(f"Feature dimensionality: {FEATURE_DIM}")
    
    image_paths = get_image_paths('../dataset')
    class_to_paths = analyze_class_distribution(image_paths)
    print_class_distribution(class_to_paths, "Original Dataset Distribution")
    
    # Apply uniform augmentation: all classes augmented by the same multiplier
    # e.g., 1.0 = no augmentation, 2.0 = double each class (preserves imbalance)
    augmentation_multiplier = 1.3  # Change to 2.0 to double all classes, etc.
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


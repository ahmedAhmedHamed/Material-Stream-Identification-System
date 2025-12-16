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
# Feature normalization
# -----------------------------

def normalize_l2(X: np.ndarray) -> np.ndarray:
    """
    L2-normalize feature vectors.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features) or 1D array
        
    Returns:
        L2-normalized feature matrix with same shape
    """
    X = np.asarray(X)
    original_shape = X.shape
    
    # Handle 1D case
    if X.ndim == 1:
        X = X.reshape(1, -1)
        was_1d = True
    else:
        was_1d = False
    
    # Compute L2 norms
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    
    # Normalize
    X_normalized = X / norms
    
    # Restore original shape if needed
    if was_1d:
        X_normalized = X_normalized.flatten()
    
    return X_normalized.astype(np.float32)


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
    
    features = features.astype(np.float32)
    # Apply L2 normalization
    features = normalize_l2(features)
    
    return features


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
    
    features = features.astype(np.float32)
    # Apply L2 normalization
    features = normalize_l2(features)
    
    return features


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


# -----------------------------
# Class-aware resampling
# -----------------------------

def calculate_target_size(class_to_paths: Dict[str, List[str]], 
                          target_size: str = 'mean') -> int:
    """
    Calculate target size for class balancing.
    
    Args:
        class_to_paths: Dictionary mapping class labels to image paths
        target_size: Strategy for target size ('mean', 'median', or int value)
        
    Returns:
        Target size as integer
    """
    class_sizes = [len(paths) for paths in class_to_paths.values()]
    
    if isinstance(target_size, (int, float)):
        return int(target_size)
    elif target_size == 'mean':
        return int(np.mean(class_sizes))
    elif target_size == 'median':
        return int(np.median(class_sizes))
    else:
        raise ValueError(f"Invalid target_size: {target_size}. Use 'mean', 'median', or an integer.")


def resample_class_paths(paths: List[str], target_size: int, 
                         random_state: Optional[int] = None) -> List[str]:
    """
    Undersample class paths to target size by random selection.
    
    Args:
        paths: List of image file paths for a class
        target_size: Target number of paths to keep
        random_state: Random seed for reproducibility
        
    Returns:
        List of sampled paths
    """
    if len(paths) <= target_size:
        return paths
    
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    return random.sample(paths, target_size)


def balance_training_set(class_to_train_paths: Dict[str, List[str]],
                        resampling_strategy: str = 'balance',
                        target_size: str = 'mean',
                        random_state: Optional[int] = None) -> Dict[str, List[str]]:
    """
    Balance training set using class-aware resampling.
    
    Args:
        class_to_train_paths: Dictionary mapping class labels to training image paths
        resampling_strategy: Strategy for resampling:
            - 'none': No resampling (return as-is)
            - 'balance': Balance all classes to target size
            - 'oversample_minority': Only oversample minority classes
            - 'undersample_majority': Only undersample majority classes
        target_size: Target size strategy ('mean', 'median', or int)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary mapping class labels to resampled image paths
    """
    if resampling_strategy == 'none':
        return class_to_train_paths
    
    # Calculate target size
    target = calculate_target_size(class_to_train_paths, target_size)
    
    balanced_paths = {}
    class_sizes = {label: len(paths) for label, paths in class_to_train_paths.items()}
    
    for label, paths in class_to_train_paths.items():
        current_size = class_sizes[label]
        
        if resampling_strategy == 'balance':
            if current_size < target:
                # Oversample: keep all original + generate augmented
                balanced_paths[label] = paths
            elif current_size > target:
                # Undersample: randomly select subset
                balanced_paths[label] = resample_class_paths(
                    paths, target, random_state
                )
            else:
                # Already at target size
                balanced_paths[label] = paths
                
        elif resampling_strategy == 'oversample_minority':
            if current_size < target:
                # Keep all original (will be augmented later)
                balanced_paths[label] = paths
            else:
                # Keep as-is
                balanced_paths[label] = paths
                
        elif resampling_strategy == 'undersample_majority':
            if current_size > target:
                # Undersample
                balanced_paths[label] = resample_class_paths(
                    paths, target, random_state
                )
            else:
                # Keep as-is
                balanced_paths[label] = paths
        else:
            raise ValueError(
                f"Invalid resampling_strategy: {resampling_strategy}. "
                f"Use 'none', 'balance', 'oversample_minority', or 'undersample_majority'."
            )
    
    return balanced_paths


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
                        resampling_strategy: str = 'none',
                        target_size: str = 'mean',
                        augmentation_multiplier: Optional[float] = None,
                        random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process training images with class-aware resampling and parallelization.
    
    Args:
        class_to_train_paths: Dictionary mapping class labels to training image paths
        resampling_strategy: Strategy for resampling:
            - 'none': No resampling (original behavior)
            - 'balance': Balance all classes to target size
            - 'oversample_minority': Only oversample minority classes
            - 'undersample_majority': Only undersample majority classes
        target_size: Target size strategy ('mean', 'median', or int) for balancing
        augmentation_multiplier: DEPRECATED - kept for backward compatibility.
                                 Use resampling_strategy instead.
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, y_train) arrays
    """
    # Handle deprecated augmentation_multiplier parameter
    if augmentation_multiplier is not None and resampling_strategy == 'none':
        # Legacy behavior: uniform augmentation
        resampling_strategy = 'none'
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
    
    # New class-aware resampling behavior
    if resampling_strategy == 'none':
        # No resampling: process all original images
        balanced_paths = class_to_train_paths
        target = None
    else:
        # Balance the training set
        balanced_paths = balance_training_set(
            class_to_train_paths, 
            resampling_strategy, 
            target_size,
            random_state
        )
        target = calculate_target_size(class_to_train_paths, target_size)
    
    X_train = []
    y_train = []
    
    for label, paths in balanced_paths.items():
        original_count = len(paths)
        
        # Process original images in parallel
        original_results = process_images_parallel(paths)
        for features, label_val in original_results:
            X_train.append(features)
            y_train.append(label_val)
        
        # Determine if oversampling is needed
        if resampling_strategy in ('balance', 'oversample_minority') and target is not None:
            if original_count < target:
                # Generate augmented samples to reach target
                needed_count = target - original_count
                augmented_results = generate_augmented_samples_parallel(
                    paths, needed_count
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
                         resampling_strategy: str = 'none',
                         target_size: str = 'mean',
                         augmentation_multiplier: Optional[float] = None):
    """
    Build feature matrix with class-aware resampling and split into train/validation.
    All extracted features are L2-normalized for better KNN/SVM performance.
    Validation dataset is 30% of original data (pre-resampling) and is not augmented.
    Training dataset uses class-aware resampling to handle class imbalance.
    
    Args:
        image_paths: list of image file paths
        train_output_path: path to save training .npz file
        val_output_path: path to save validation .npz file
        test_size: proportion of original data for validation (default: 0.3)
        random_state: random seed for reproducibility (default: 42)
        resampling_strategy: Strategy for handling class imbalance:
            - 'none': No resampling (original behavior)
            - 'balance': Balance all classes to target size (oversample minority, undersample majority)
            - 'oversample_minority': Only oversample minority classes to target size
            - 'undersample_majority': Only undersample majority classes to target size
        target_size: Target size for balancing ('mean', 'median', or int value)
        augmentation_multiplier: DEPRECATED - kept for backward compatibility.
                                 Use resampling_strategy='none' with augmentation_multiplier
                                 for legacy uniform augmentation behavior.
        
    Returns:
        tuple of (X_train, y_train, X_val, y_val) arrays
        All features are L2-normalized
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
        resampling_strategy=resampling_strategy,
        target_size=target_size,
        augmentation_multiplier=augmentation_multiplier,
        random_state=random_state
    )
    
    # Features are already L2-normalized in extract_features functions
    # Save to files
    np.savez(train_output_path, X=X_train, y=y_train)
    np.savez(val_output_path, X=X_val, y=y_val)
    
    return X_train, y_train, X_val, y_val


# -----------------------------
# Feature scaling
# -----------------------------

def scale_features(X_train, X_val=None):
    """
    Standardize features (required for SVM / k-NN).
    
    Note: Features are already L2-normalized during extraction.
    This function applies StandardScaler (zero mean, unit variance)
    which works correctly with L2-normalized features.
    
    Args:
        X_train: Training feature matrix (already L2-normalized)
        X_val: Optional validation feature matrix (already L2-normalized)
        
    Returns:
        If X_val is provided: (X_train_scaled, X_val_scaled, scaler)
        Otherwise: (X_train_scaled, scaler)
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
    
    # Class-aware resampling to handle imbalance
    # Options: 'none', 'balance', 'oversample_minority', 'undersample_majority'
    resampling_strategy = 'balance'  # Balance all classes to mean size
    target_size = 'mean'  # Use mean class size as target
    
    X_train, y_train, X_val, y_val = build_feature_matrix(
        image_paths,
        resampling_strategy=resampling_strategy,
        target_size=target_size
    )
    
    print(f"\n{'='*60}")
    print("Feature Extraction Complete")
    print(f"{'='*60}")
    print(f"Resampling strategy: {resampling_strategy}")
    print(f"Target size: {target_size}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features are L2-normalized: Yes")
    
    # Show final training distribution
    from collections import Counter
    train_dist = Counter(y_train)
    print(f"\nFinal training distribution:")
    for label, count in sorted(train_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label:15s}: {count:4d} samples")


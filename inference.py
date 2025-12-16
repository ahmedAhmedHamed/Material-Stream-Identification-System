"""
inference.py

Inference script for material recognition using trained classifiers.
Extracts features from new images and makes predictions using saved KNN or SVM models.

Features:
- Single image and batch prediction
- Automatic scaler discovery
- Support for both KNN and SVM classifiers
- Lightweight: only requires classifier and scaler files

Author: (your name)
"""
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Import custom classifier classes for proper unpickling
from classifiers.KNN import WeightedKNeighborsClassifier

from vector_extraction.deep_feature_extractor import (
    extract_features,
    extract_features_batch,
    load_image
)


# -----------------------------
# Path Resolution
# -----------------------------

def _resolve_file_path(file_path: str) -> Path:
    """
    Resolve file path, checking both absolute and relative locations.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Resolved Path object
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    
    if path.exists():
        return path
    
    abs_path = Path(__file__).parent / path
    if abs_path.exists():
        return abs_path
    
    raise FileNotFoundError(f"Could not find file: {file_path}")


# -----------------------------
# Model and Scaler Loading
# -----------------------------

def find_scaler_path(classifier_path: str) -> Path:
    """
    Automatically find scaler path from classifier directory.
    
    Args:
        classifier_path: Path to classifier .joblib file
        
    Returns:
        Path to scaler file in the same directory
        
    Raises:
        FileNotFoundError: If scaler file not found
    """
    classifier_path_obj = _resolve_file_path(classifier_path)
    classifier_dir = classifier_path_obj.parent
    
    # Try common scaler names
    scaler_names = [
        'knn_scaler.joblib',
        'svm_scaler.joblib',
        'scaler.joblib'
    ]
    
    for scaler_name in scaler_names:
        scaler_path = classifier_dir / scaler_name
        if scaler_path.exists():
            return scaler_path
    
    # If not found, raise error
    raise FileNotFoundError(
        f"Could not find scaler file in {classifier_dir}. "
        f"Expected one of: {scaler_names}"
    )


def load_model_and_scaler(classifier_path: str, 
                         scaler_path: Optional[str] = None) -> Tuple[object, object]:
    """
    Load trained classifier and scaler from files.
    
    Args:
        classifier_path: Path to classifier .joblib file
        scaler_path: Optional path to scaler .joblib file.
                    If None, automatically searches in classifier directory.
        
    Returns:
        Tuple of (classifier, scaler)
    """
    # Register custom classes for proper unpickling
    # This handles cases where the classifier was saved from __main__
    import sys
    
    # Ensure WeightedKNeighborsClassifier is available in __main__ namespace
    # This is needed when the classifier was saved from a script run as __main__
    try:
        from classifiers.KNN import WeightedKNeighborsClassifier
        if '__main__' in sys.modules:
            main_module = sys.modules['__main__']
            if not hasattr(main_module, 'WeightedKNeighborsClassifier'):
                setattr(main_module, 'WeightedKNeighborsClassifier', 
                       WeightedKNeighborsClassifier)
    except ImportError:
        pass  # Class not available, will fail during unpickling if needed
    
    classifier_path_obj = _resolve_file_path(classifier_path)
    classifier = joblib.load(classifier_path_obj)
    
    if scaler_path is None:
        scaler_path_obj = find_scaler_path(classifier_path)
    else:
        scaler_path_obj = _resolve_file_path(scaler_path)
    
    scaler = joblib.load(scaler_path_obj)
    
    return classifier, scaler


# -----------------------------
# Feature Extraction and Scaling
# -----------------------------

def extract_and_scale_features(image: np.ndarray, 
                               scaler: object) -> np.ndarray:
    """
    Extract features from image and apply scaling.
    
    Args:
        image: RGB uint8 image (H, W, C)
        scaler: Fitted StandardScaler
        
    Returns:
        Scaled feature vector (1D array)
    """
    features = extract_features(image)
    features_2d = features.reshape(1, -1)
    features_scaled = scaler.transform(features_2d)
    return features_scaled[0]


def extract_and_scale_features_batch(images: List[np.ndarray],
                                     scaler: object) -> np.ndarray:
    """
    Extract features from batch of images and apply scaling.
    
    Args:
        images: List of RGB uint8 images
        scaler: Fitted StandardScaler
        
    Returns:
        Scaled feature matrix (2D array, shape: [n_images, n_features])
    """
    features = extract_features_batch(images)
    features_scaled = scaler.transform(features)
    return features_scaled


# -----------------------------
# Prediction Functions
# -----------------------------

def predict_image(image_path: str,
                  classifier_path: str,
                  scaler_path: Optional[str] = None) -> str:
    """
    Predict class label for a single image.
    
    Args:
        image_path: Path to image file
        classifier_path: Path to classifier .joblib file
        scaler_path: Optional path to scaler .joblib file.
                    If None, automatically searches in classifier directory.
        
    Returns:
        Predicted class label
    """
    classifier, scaler = load_model_and_scaler(classifier_path, scaler_path)
    image = load_image(image_path)
    features_scaled = extract_and_scale_features(image, scaler)
    prediction = classifier.predict(features_scaled.reshape(1, -1))
    return prediction[0]


def predict_batch(image_paths: List[str],
                  classifier_path: str,
                  scaler_path: Optional[str] = None) -> List[str]:
    """
    Predict class labels for a batch of images.
    
    Args:
        image_paths: List of paths to image files
        classifier_path: Path to classifier .joblib file
        scaler_path: Optional path to scaler .joblib file.
                    If None, automatically searches in classifier directory.
        
    Returns:
        List of predicted class labels
    """
    classifier, scaler = load_model_and_scaler(classifier_path, scaler_path)
    
    images = []
    valid_paths = []
    for path in image_paths:
        try:
            image = load_image(path)
            images.append(image)
            valid_paths.append(path)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            continue
    
    if not images:
        raise ValueError("No valid images could be loaded")
    
    features_scaled = extract_and_scale_features_batch(images, scaler)
    predictions = classifier.predict(features_scaled)
    
    return predictions.tolist()


def predict_with_probabilities(image_path: str,
                              classifier_path: str,
                              scaler_path: Optional[str] = None) -> Tuple[str, Dict[str, float]]:
    """
    Predict class label with probabilities for all classes.
    
    Args:
        image_path: Path to image file
        classifier_path: Path to classifier .joblib file
        scaler_path: Optional path to scaler .joblib file.
                    If None, automatically searches in classifier directory.
        
    Returns:
        Tuple of (predicted_class, probabilities_dict)
        probabilities_dict maps class names to their probability scores
        
    Raises:
        AttributeError: If classifier doesn't support predict_proba
        RuntimeError: If predict_proba fails for any reason
    """
    classifier, scaler = load_model_and_scaler(classifier_path, scaler_path)
    image = load_image(image_path)
    features_scaled = extract_and_scale_features(image, scaler)
    features_2d = features_scaled.reshape(1, -1)
    
    prediction = classifier.predict(features_2d)[0]
    
    # Get probabilities for all classes - fail if not available
    if not hasattr(classifier, 'predict_proba'):
        raise AttributeError(
            f"Classifier {type(classifier).__name__} does not support predict_proba(). "
            f"Cannot get probabilities."
        )
    
    try:
        probabilities = classifier.predict_proba(features_2d)[0]
        classes = classifier.classes_
        probabilities_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
    except Exception as e:
        raise RuntimeError(
            f"Failed to get probabilities from classifier: {e}"
        ) from e
    
    return prediction, probabilities_dict


def predict_with_confidence(image_path: str,
                            classifier_path: str,
                            scaler_path: Optional[str] = None) -> Tuple[str, float]:
    """
    Predict class label with confidence score (maximum probability) for a single image.
    
    Args:
        image_path: Path to image file
        classifier_path: Path to classifier .joblib file
        scaler_path: Optional path to scaler .joblib file.
                    If None, automatically searches in classifier directory.
        
    Returns:
        Tuple of (predicted_class, confidence_score)
        Confidence is between 0 and 1 (higher is more confident)
        
    Raises:
        AttributeError: If classifier doesn't support predict_proba
        RuntimeError: If predict_proba fails for any reason
    """
    prediction, probabilities = predict_with_probabilities(image_path, classifier_path, scaler_path)
    confidence = max(probabilities.values())
    return prediction, confidence


# -----------------------------
# Example Usage
# -----------------------------

def main():
    """
    Example usage of inference functions.
    """
    print("=" * 60)
    print("Material Recognition Inference")
    print("=" * 60)
    
    # Example: Using KNN classifier
    classifier_path = "classifiers/KNN_best/knn_classifier.joblib"
    
    # Check if classifier exists
    try:
        classifier_path_obj = _resolve_file_path(classifier_path)
        print(f"\nUsing classifier: {classifier_path_obj}")
    except FileNotFoundError:
        print(f"\nError: Classifier not found at {classifier_path}")
        print("Please train a classifier first or update the path.")
        return
    
    # Example: Single image prediction
    print("\n" + "-" * 60)
    print("Example 1: Single Image Prediction")
    print("-" * 60)
    
    # Get a sample image from dataset (if available)
    dataset_path = Path("dataset")
    sample_images = []
    if dataset_path.exists():
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir():
                for img_file in class_dir.glob("*.jpg"):
                    sample_images.append(str(img_file))
                    break
                if sample_images:
                    break
    
    if sample_images:
        image_path = sample_images[0]
        print(f"Predicting: {image_path}")
        try:
            prediction = predict_image(image_path, classifier_path)
            print(f"Prediction: {prediction}")
            
            # With probabilities
            pred, probs = predict_with_probabilities(image_path, classifier_path)
            print(f"\nPrediction: {pred}")
            print("Probabilities:")
            for class_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
        except Exception as e:
            print(f"Error during prediction: {e}")
    else:
        print("No sample images found. Update image_path to test inference.")
    
    # Example: Batch prediction
    print("\n" + "-" * 60)
    print("Example 2: Batch Prediction")
    print("-" * 60)
    
    if len(sample_images) > 1:
        batch_paths = sample_images[:3]  # Use first 3 images
        print(f"Predicting {len(batch_paths)} images...")
        try:
            predictions = predict_batch(batch_paths, classifier_path)
            for img_path, pred in zip(batch_paths, predictions):
                print(f"  {Path(img_path).name}: {pred}")
        except Exception as e:
            print(f"Error during batch prediction: {e}")
    else:
        print("Not enough sample images for batch prediction.")
    
    print("\n" + "=" * 60)
    print("Inference examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


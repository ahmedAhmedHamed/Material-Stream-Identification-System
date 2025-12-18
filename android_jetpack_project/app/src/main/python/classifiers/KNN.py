"""
KNN Classifier for Material Recognition

K-nearest neighbors classifier using sklearn that loads features from
features_train.npz and features_val.npz, scales features, fits the model,
and saves it for future use.

Author: (your name)
"""
import os
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from vector_extraction.feature_extractor import scale_features


# -----------------------------
# Data Loading
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
    
    abs_path = Path(__file__).parent.parent / path
    if abs_path.exists():
        return abs_path
    
    raise FileNotFoundError(f"Could not find file: {file_path}")


def load_train_features(file_path: str = "../features_train.npz") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training features from npz file.
    
    Args:
        file_path: Path to features_train.npz file
        
    Returns:
        Tuple of (X_train, y_train) arrays
    """
    path = _resolve_file_path(file_path)
    data = np.load(path)
    
    if 'X' not in data or 'y' not in data:
        raise ValueError(f"File {file_path} must contain 'X' and 'y' arrays")
    
    return data['X'], data['y']


def load_val_features(file_path: str = "../features_val.npz") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load validation features from npz file.
    
    Args:
        file_path: Path to features_val.npz file
        
    Returns:
        Tuple of (X_val, y_val) arrays
    """
    path = _resolve_file_path(file_path)
    data = np.load(path)
    
    if 'X' not in data or 'y' not in data:
        raise ValueError(f"File {file_path} must contain 'X' and 'y' arrays")
    
    return data['X'], data['y']


# -----------------------------
# Class Weighting
# -----------------------------

def compute_class_weights(y_train: np.ndarray, class_weight: str = 'balanced') -> Dict[Any, float]:
    """
    Compute class weights based on class frequencies.
    
    Args:
        y_train: Training labels
        class_weight: Class weight mode (default: 'balanced')
                      'balanced' automatically weights classes inversely
                      proportional to their frequency
        
    Returns:
        Dictionary mapping class labels to weights
    """
    if class_weight == 'balanced':
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        n_samples = len(y_train)
        n_classes = len(unique_classes)
        
        # Compute weights: n_samples / (n_classes * class_count)
        class_weights = n_samples / (n_classes * class_counts)
        
        return dict(zip(unique_classes, class_weights))
    else:
        # No weighting - return uniform weights
        unique_classes = np.unique(y_train)
        return {cls: 1.0 for cls in unique_classes}


class WeightedKNeighborsClassifier(KNeighborsClassifier):
    """
    KNN classifier with class weighting support.
    
    Wraps KNeighborsClassifier and applies class weights during prediction
    by modifying the voting mechanism to weight votes by class frequency.
    """
    
    def __init__(self, n_neighbors: int = 5, class_weights: Optional[Dict[Any, float]] = None, **kwargs):
        """
        Initialize weighted KNN classifier.
        
        Args:
            n_neighbors: Number of neighbors
            class_weights: Dictionary mapping class labels to weights
            **kwargs: Additional arguments passed to KNeighborsClassifier
        """
        super().__init__(n_neighbors=n_neighbors, **kwargs)
        self.class_weights = class_weights
        self._y_train = None
    
    def fit(self, X, y):
        """
        Fit the classifier and store training labels.
        
        Args:
            X: Training feature matrix
            y: Training labels
        """
        super().fit(X, y)
        self._y_train = y
        return self
    
    def predict(self, X):
        """
        Predict class labels with class-weighted voting.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted class labels
        """
        if self.class_weights is None or self._y_train is None:
            return super().predict(X)
        
        # Get distances and indices of nearest neighbors
        distances, indices = self.kneighbors(X)
        
        # Get class labels of neighbors
        neighbor_classes = self._y_train[indices]
        
        # Apply class weights to votes
        predictions = []
        for i in range(len(X)):
            neighbor_votes = neighbor_classes[i]
            neighbor_dists = distances[i]
            
            # Count weighted votes for each class
            weighted_votes = {}
            for j, cls in enumerate(neighbor_votes):
                # Get base class weight
                class_weight = self.class_weights.get(cls, 1.0)
                
                # Apply distance weighting if enabled
                if self.weights == 'distance':
                    # Use inverse distance as weight
                    dist_weight = 1.0 / (neighbor_dists[j] + 1e-10)
                    vote_weight = class_weight * dist_weight
                else:
                    # Uniform weighting
                    vote_weight = class_weight
                
                weighted_votes[cls] = weighted_votes.get(cls, 0.0) + vote_weight
            
            # Predict class with highest weighted vote
            if weighted_votes:
                predictions.append(max(weighted_votes, key=weighted_votes.get))
            else:
                # Fallback to standard prediction
                predictions.append(super().predict(X[i:i+1])[0])
        
        return np.array(predictions)


# -----------------------------
# Model Training
# -----------------------------

def train_knn_classifier(X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: Optional[np.ndarray] = None,
                        n_neighbors: int = 5,
                        class_weight: str = 'balanced') -> Tuple[KNeighborsClassifier, object]:
    """
    Train KNN classifier on scaled features.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        X_val: Optional validation feature matrix for scaling
        n_neighbors: Number of neighbors for KNN (default: 5)
        class_weight: Class weight mode (default: 'balanced')
                      'balanced' automatically weights classes inversely
                      proportional to their frequency
        
    Returns:
        Tuple of (trained_classifier, scaler)
    """
    X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)
    
    # Compute class weights if class_weight is enabled
    class_weights = compute_class_weights(y_train, class_weight) if class_weight == 'balanced' else None
    
    # Use weighted classifier if class weights are provided
    if class_weights is not None:
        classifier = WeightedKNeighborsClassifier(n_neighbors=n_neighbors, class_weights=class_weights)
    else:
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    classifier.fit(X_train_scaled, y_train)
    
    return classifier, scaler


# -----------------------------
# Model Evaluation
# -----------------------------

def get_evaluation_metrics(classifier: KNeighborsClassifier,
                          scaler: object,
                          X_val: np.ndarray,
                          y_val: np.ndarray,
                          class_weight: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate classifier and return metrics as dictionary.
    
    Args:
        classifier: Trained KNN classifier
        scaler: Fitted StandardScaler
        X_val: Validation feature matrix
        y_val: Validation labels
        class_weight: Original class_weight parameter used (optional)
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    X_val_scaled = scaler.transform(X_val)
    y_pred = classifier.predict(X_val_scaled)
    
    unique_classes = np.unique(y_val).tolist()
    overall_accuracy = float(accuracy_score(y_val, y_pred))
    
    per_class_accuracy = {}
    for cls in unique_classes:
        mask = y_val == cls
        if mask.sum() > 0:
            class_acc = float(accuracy_score(y_val[mask], y_pred[mask]))
            per_class_accuracy[str(cls)] = class_acc
    
    cm = confusion_matrix(y_val, y_pred, labels=unique_classes)
    confusion_matrix_dict = {
        "labels": [str(cls) for cls in unique_classes],
        "matrix": cm.tolist()
    }
    
    report_dict = classification_report(
        y_val, y_pred, target_names=[str(c) for c in unique_classes], output_dict=True
    )
    
    metrics = {
        "overall_accuracy": overall_accuracy,
        "overall_accuracy_percent": overall_accuracy * 100,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion_matrix_dict,
        "classification_report": report_dict,
        "n_neighbors": classifier.n_neighbors,
        "validation_samples": int(len(y_val))
    }
    
    if class_weight is not None:
        metrics["class_weight"] = class_weight
    
    return metrics


def evaluate_classifier(classifier: KNeighborsClassifier,
                       scaler: object,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       class_weight: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate classifier, print results, and return metrics.
    
    Args:
        classifier: Trained KNN classifier
        scaler: Fitted StandardScaler
        X_val: Validation feature matrix
        y_val: Validation labels
        class_weight: Original class_weight parameter used (optional)
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    metrics = get_evaluation_metrics(classifier, scaler, X_val, y_val, class_weight=class_weight)
    
    print(f"\n{'='*60}")
    print("CLASSIFIER EVALUATION RESULTS")
    print(f"{'='*60}")
    
    print(f"\n1. Overall Accuracy: {metrics['overall_accuracy']:.4f} "
          f"({metrics['overall_accuracy_percent']:.2f}%)")
    
    print(f"\n2. Per-Class Accuracy:")
    for cls, acc in metrics['per_class_accuracy'].items():
        print(f"   {cls}: {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\n3. Confusion Matrix:")
    labels = metrics['confusion_matrix']['labels']
    cm = metrics['confusion_matrix']['matrix']
    print("   Predicted ->")
    header = "   " + " ".join([f"{cls:>10}" for cls in labels])
    print(header)
    for i, cls in enumerate(labels):
        row = f"   {cls:>10} " + " ".join([f"{val:>10}" for val in cm[i]])
        print(row)
    
    print(f"\n4. Detailed Classification Report:")
    report_str = classification_report(
        y_val, 
        classifier.predict(scaler.transform(X_val)),
        target_names=labels
    )
    print(report_str)
    
    return metrics


# -----------------------------
# Directory Management
# -----------------------------

def get_next_knn_directory(base_path: str = ".") -> Path:
    """
    Find next available KNN directory name.
    
    Args:
        base_path: Base directory to search in
        
    Returns:
        Path to next available KNN directory
    """
    base = Path(base_path)
    number = 0
    
    while True:
        dir_name = base / f"KNN_{number}"
        if not dir_name.exists():
            return dir_name
        number += 1


# -----------------------------
# Model Persistence
# -----------------------------

def save_classifier(classifier: KNeighborsClassifier,
                   scaler: object,
                   metrics: Dict[str, Any],
                   base_path: str = "."):
    """
    Save classifier, scaler, and metrics to numbered directory.
    
    Args:
        classifier: Trained KNN classifier
        scaler: Fitted StandardScaler
        metrics: Evaluation metrics dictionary
        base_path: Base directory to save in
    """
    output_dir = get_next_knn_directory(base_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    classifier_path = output_dir / "knn_classifier.joblib"
    scaler_path = output_dir / "knn_scaler.joblib"
    metrics_path = output_dir / "evaluation_metrics.json"
    
    joblib.dump(classifier, classifier_path)
    joblib.dump(scaler, scaler_path)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nClassifier saved to {classifier_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Evaluation metrics saved to {metrics_path}")
    print(f"All files saved in directory: {output_dir}")


def update_accuracies_json(n: int, accuracy: float, json_path: str = "knn_accuracies.json"):
    """
    Update accuracies JSON file with new n and accuracy pair.
    
    Args:
        n: Number of neighbors
        accuracy: Overall accuracy value
        json_path: Path to JSON file to update
    """
    path = Path(json_path)
    
    if path.exists():
        with open(path, 'r') as f:
            data = json.load(f)
    else:
        data = {"accuracies": []}
    
    data["accuracies"].append({
        "n": n,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100
    })
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def save_best_model(classifier: KNeighborsClassifier,
                   scaler: object,
                   metrics: Dict[str, Any],
                   base_path: str = "."):
    """
    Save best model to a dedicated directory.
    
    Args:
        classifier: Trained KNN classifier
        scaler: Fitted StandardScaler
        metrics: Evaluation metrics dictionary
        base_path: Base directory to save in
    """
    output_dir = Path(base_path) / "KNN_best"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    classifier_path = output_dir / "knn_classifier.joblib"
    scaler_path = output_dir / "knn_scaler.joblib"
    metrics_path = output_dir / "evaluation_metrics.json"
    
    joblib.dump(classifier, classifier_path)
    joblib.dump(scaler, scaler_path)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print("BEST MODEL SAVED")
    print(f"{'='*60}")
    print(f"Best model saved to {classifier_path}")
    print(f"Best scaler saved to {scaler_path}")
    print(f"Best metrics saved to {metrics_path}")
    print(f"Best N value: {metrics.get('n_neighbors', 'N/A')}")
    print(f"Best accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy_percent']:.2f}%)")


# -----------------------------
# Main Execution
# -----------------------------

if __name__ == "__main__":
    print("Loading training features...")
    X_train, y_train = load_train_features()
    print(f"Training samples: {len(X_train)}")
    
    print("Loading validation features...")
    X_val, y_val = load_val_features()
    print(f"Validation samples: {len(X_val)}")
    
    best_accuracy = -1.0
    best_classifier = None
    best_scaler = None
    best_metrics = None
    best_n = None
    
    n_values = list(range(1, 51, 5))  # n = 1, 6, 11, 16, 21, 26, 31, 36, 41, 46
    
    print(f"\n{'='*60}")
    print(f"TRAINING KNN CLASSIFIERS FOR N = {n_values}")
    print(f"{'='*60}\n")
    
    for n in n_values:
        print(f"\n{'='*60}")
        print(f"Training KNN with n_neighbors = {n}")
        print(f"{'='*60}")
        
        classifier, scaler = train_knn_classifier(
            X_train, y_train, X_val, n_neighbors=n, class_weight='balanced'
        )
        
        metrics = get_evaluation_metrics(
            classifier, scaler, X_val, y_val, class_weight='balanced'
        )
        accuracy = metrics['overall_accuracy']
        
        print(f"N = {n}: Accuracy = {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        update_accuracies_json(n, accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_classifier = classifier
            best_scaler = scaler
            best_metrics = metrics
            best_n = n
            print(f"*** New best model! Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%) ***")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best N value: {best_n}")
    print(f"Best accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    if best_classifier is not None:
        print("\nSaving best model...")
        save_best_model(best_classifier, best_scaler, best_metrics)
    
    print("\nKNN classifier training complete!")


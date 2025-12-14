"""
SVM Classifier for Material Recognition

Support Vector Machine classifier using sklearn that loads features from
features_train.npz and features_val.npz, scales features, fits the model,
and saves it for future use.

Author: (your name)
"""
import os
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
from sklearn.svm import SVC
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
# Model Training
# -----------------------------

def train_svm_classifier(X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: Optional[np.ndarray] = None,
                        C: float = 1.0,
                        kernel: str = 'rbf',
                        gamma: str = 'scale') -> Tuple[SVC, object]:
    """
    Train SVM classifier on scaled features.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        X_val: Optional validation feature matrix for scaling
        C: Regularization parameter (default: 1.0)
        kernel: Kernel type (default: 'rbf')
        gamma: Kernel coefficient (default: 'scale')
        
    Returns:
        Tuple of (trained_classifier, scaler)
    """
    X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)
    
    classifier = SVC(C=C, kernel=kernel, gamma=gamma)
    classifier.fit(X_train_scaled, y_train)
    
    return classifier, scaler


# -----------------------------
# Model Evaluation
# -----------------------------

def get_evaluation_metrics(classifier: SVC,
                          scaler: object,
                          X_val: np.ndarray,
                          y_val: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate classifier and return metrics as dictionary.
    
    Args:
        classifier: Trained SVM classifier
        scaler: Fitted StandardScaler
        X_val: Validation feature matrix
        y_val: Validation labels
        
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
    
    return {
        "overall_accuracy": overall_accuracy,
        "overall_accuracy_percent": overall_accuracy * 100,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion_matrix_dict,
        "classification_report": report_dict,
        "model_parameters": {
            "C": float(classifier.C),
            "kernel": str(classifier.kernel),
            "gamma": str(classifier.gamma)
        },
        "validation_samples": int(len(y_val))
    }


def evaluate_classifier(classifier: SVC,
                       scaler: object,
                       X_val: np.ndarray,
                       y_val: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate classifier, print results, and return metrics.
    
    Args:
        classifier: Trained SVM classifier
        scaler: Fitted StandardScaler
        X_val: Validation feature matrix
        y_val: Validation labels
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    metrics = get_evaluation_metrics(classifier, scaler, X_val, y_val)
    
    print(f"\n{'='*60}")
    print("SVM CLASSIFIER EVALUATION RESULTS")
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

def get_next_svm_directory(base_path: str = ".") -> Path:
    """
    Find next available SVM directory name.
    
    Args:
        base_path: Base directory to search in
        
    Returns:
        Path to next available SVM directory
    """
    base = Path(base_path)
    number = 0
    
    while True:
        dir_name = base / f"SVM_{number}"
        if not dir_name.exists():
            return dir_name
        number += 1


# -----------------------------
# Model Persistence
# -----------------------------

def save_classifier(classifier: SVC,
                   scaler: object,
                   metrics: Dict[str, Any],
                   base_path: str = "."):
    """
    Save classifier, scaler, and metrics to numbered directory.
    
    Args:
        classifier: Trained SVM classifier
        scaler: Fitted StandardScaler
        metrics: Evaluation metrics dictionary
        base_path: Base directory to save in
    """
    output_dir = get_next_svm_directory(base_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    classifier_path = output_dir / "svm_classifier.joblib"
    scaler_path = output_dir / "svm_scaler.joblib"
    metrics_path = output_dir / "evaluation_metrics.json"
    
    joblib.dump(classifier, classifier_path)
    joblib.dump(scaler, scaler_path)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nClassifier saved to {classifier_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Evaluation metrics saved to {metrics_path}")
    print(f"All files saved in directory: {output_dir}")


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
    
    print("Scaling features and training SVM classifier...")
    classifier, scaler = train_svm_classifier(X_train, y_train, X_val, C=1.0, kernel='rbf', gamma='scale')
    
    print("Evaluating classifier...")
    metrics = evaluate_classifier(classifier, scaler, X_val, y_val)
    
    print("\nSaving classifier, scaler, and evaluation metrics...")
    save_classifier(classifier, scaler, metrics)
    
    print("\nSVM classifier training complete!")

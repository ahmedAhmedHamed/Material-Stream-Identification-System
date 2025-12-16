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
from itertools import product

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
                        gamma: str = 'scale',
                        class_weight: str = 'balanced') -> Tuple[SVC, object]:
    """
    Train SVM classifier on scaled features.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        X_val: Optional validation feature matrix for scaling
        C: Regularization parameter (default: 1.0)
        kernel: Kernel type (default: 'rbf')
        gamma: Kernel coefficient (default: 'scale')
        class_weight: Class weight mode (default: 'balanced')
                      'balanced' automatically weights classes inversely
                      proportional to their frequency
        
    Returns:
        Tuple of (trained_classifier, scaler)
    """
    X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)
    
    classifier = SVC(C=C, kernel=kernel, gamma=gamma, class_weight=class_weight)
    classifier.fit(X_train_scaled, y_train)
    
    return classifier, scaler


# -----------------------------
# Model Evaluation
# -----------------------------

def get_evaluation_metrics(classifier: SVC,
                          scaler: object,
                          X_val: np.ndarray,
                          y_val: np.ndarray,
                          C: Optional[float] = None,
                          kernel: Optional[str] = None,
                          gamma: Optional[Any] = None,
                          class_weight: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate classifier and return metrics as dictionary.
    
    Args:
        classifier: Trained SVM classifier
        scaler: Fitted StandardScaler
        X_val: Validation feature matrix
        y_val: Validation labels
        C: Original C parameter used (optional)
        kernel: Original kernel parameter used (optional)
        gamma: Original gamma parameter used (optional)
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
    
    # Use provided parameters or fall back to classifier attributes
    model_params = {
        "C": float(C if C is not None else classifier.C),
        "kernel": str(kernel if kernel is not None else classifier.kernel),
        "gamma": str(gamma if gamma is not None else classifier.gamma),
        "class_weight": str(class_weight if class_weight is not None else classifier.class_weight)
    }
    
    return {
        "overall_accuracy": overall_accuracy,
        "overall_accuracy_percent": overall_accuracy * 100,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion_matrix_dict,
        "classification_report": report_dict,
        "model_parameters": model_params,
        "validation_samples": int(len(y_val))
    }


def evaluate_classifier(classifier: SVC,
                       scaler: object,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       C: Optional[float] = None,
                       kernel: Optional[str] = None,
                       gamma: Optional[Any] = None,
                       class_weight: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate classifier, print results, and return metrics.
    
    Args:
        classifier: Trained SVM classifier
        scaler: Fitted StandardScaler
        X_val: Validation feature matrix
        y_val: Validation labels
        C: Original C parameter used (optional)
        kernel: Original kernel parameter used (optional)
        gamma: Original gamma parameter used (optional)
        class_weight: Original class_weight parameter used (optional)
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    metrics = get_evaluation_metrics(
        classifier, scaler, X_val, y_val, C=C, kernel=kernel, gamma=gamma, class_weight=class_weight
    )
    
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

def save_metrics_json(metrics: Dict[str, Any],
                     C: float,
                     kernel: str,
                     gamma: Any,
                     base_path: str = ".") -> Path:
    """
    Save metrics JSON for a single model configuration.
    
    Args:
        metrics: Evaluation metrics dictionary
        C: C parameter value
        kernel: Kernel type
        gamma: Gamma parameter value
        base_path: Base directory to save in
        
    Returns:
        Path to saved metrics file
    """
    results_dir = Path(base_path) / "grid_search_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create descriptive filename
    gamma_str = str(gamma).replace('.', '_')
    filename = f"metrics_C{C}_kernel{kernel}_gamma{gamma_str}.json"
    metrics_path = results_dir / filename
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics_path


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
    
    # Define parameter grid for grid search
    C_values = [0.1, 1, 10, 100]
    # C_values = [0.1]
    gamma_values = ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    # gamma_values = ['scale']
    kernel_values = ['rbf', 'linear', 'poly']
    # kernel_values = ['rbf']

    # Create grid search results directory
    results_dir = Path("grid_search_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("GRID SEARCH: Testing all hyperparameter combinations")
    print(f"{'='*60}")
    print(f"Total combinations: {len(C_values) * len(gamma_values) * len(kernel_values)}")
    print(f"C values: {C_values}")
    print(f"Gamma values: {gamma_values}")
    print(f"Kernel values: {kernel_values}")
    print(f"Results will be saved to: {results_dir}")
    print(f"{'='*60}\n")
    
    best_accuracy = 0.0
    best_classifier = None
    best_scaler = None
    best_metrics = None
    best_params = None
    combination_num = 0
    
    # Iterate over all parameter combinations
    for C, gamma, kernel in product(C_values, gamma_values, kernel_values):
        combination_num += 1
        
        # Skip invalid combinations (gamma not applicable for linear kernel)
        if kernel == 'linear' and gamma != 'scale':
            continue
        
        print(f"\n[{combination_num}] Testing: C={C}, kernel={kernel}, gamma={gamma}")
        
        try:
            # Train classifier with current parameters
            classifier, scaler = train_svm_classifier(
                X_train, y_train, X_val, C=C, kernel=kernel, gamma=gamma, class_weight='balanced'
            )
            
            # Evaluate classifier with original parameters
            metrics = get_evaluation_metrics(
                classifier, scaler, X_val, y_val, C=C, kernel=kernel, gamma=gamma, class_weight='balanced'
            )
            accuracy = metrics['overall_accuracy']
            
            print(f"    Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Save metrics JSON for this model
            metrics_path = save_metrics_json(metrics, C, kernel, gamma)
            print(f"    Metrics saved to: {metrics_path.name}")
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_classifier = classifier
                best_scaler = scaler
                best_metrics = metrics
                best_params = {'C': C, 'kernel': kernel, 'gamma': gamma, 'class_weight': 'balanced'}
                print(f"    *** NEW BEST! ***")
        
        except Exception as e:
            print(f"    ERROR: {str(e)}")
            continue
    
    # Print summary and save best model
    print(f"\n{'='*60}")
    print("GRID SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Best accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"Best parameters: {best_params}")
    print(f"All metrics JSON files saved to: {results_dir}")
    print(f"{'='*60}\n")
    
    if best_classifier is not None:
        print("Evaluating best classifier...")
        evaluate_classifier(
            best_classifier, best_scaler, X_val, y_val,
            C=best_params['C'], kernel=best_params['kernel'], gamma=best_params['gamma'],
            class_weight=best_params['class_weight']
        )
        
        print("\nSaving best classifier, scaler, and evaluation metrics...")
        save_classifier(best_classifier, best_scaler, best_metrics)
        
        print("\nSVM classifier training complete!")
    else:
        print("ERROR: No valid model was trained!")



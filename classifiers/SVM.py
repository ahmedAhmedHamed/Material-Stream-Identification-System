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
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, balanced_accuracy_score, precision_score, recall_score
)
from collections import Counter
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
# Class Weight Utilities
# -----------------------------

def get_class_distribution(y: np.ndarray) -> Dict[str, Any]:
    """
    Analyze class distribution in dataset.
    
    Args:
        y: Array of class labels
        
    Returns:
        Dictionary containing class counts and statistics
    """
    class_counts = Counter(y)
    unique_classes = sorted(class_counts.keys())
    counts = [class_counts[cls] for cls in unique_classes]
    
    return {
        "class_counts": {str(cls): int(count) for cls, count in class_counts.items()},
        "unique_classes": [str(cls) for cls in unique_classes],
        "total_samples": int(len(y)),
        "num_classes": int(len(unique_classes)),
        "min_count": int(min(counts)),
        "max_count": int(max(counts)),
        "mean_count": float(np.mean(counts)),
        "imbalance_ratio": float(max(counts) / min(counts)) if min(counts) > 0 else float('inf')
    }


def compute_class_weights(y_train: np.ndarray, 
                         strategy: str = 'balanced') -> Optional[Dict[str, float]]:
    """
    Compute class weights using different strategies.
    
    Args:
        y_train: Training labels
        strategy: Weight computation strategy
                 - 'balanced': sklearn's balanced weights (inverse frequency)
                 - 'inverse_freq': Simple inverse frequency
                 - None: Return None (no weighting)
                 
    Returns:
        Dictionary mapping class labels to weights, or None
    """
    if strategy is None:
        return None
    
    if strategy == 'balanced':
        # Use sklearn's compute_class_weight logic
        from sklearn.utils.class_weight import compute_class_weight
        unique_classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
        return {str(cls): float(weight) for cls, weight in zip(unique_classes, weights)}
    
    elif strategy == 'inverse_freq':
        class_counts = Counter(y_train)
        total = len(y_train)
        num_classes = len(class_counts)
        weights = {}
        for cls, count in class_counts.items():
            weights[str(cls)] = float(total / (num_classes * count))
        return weights
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'balanced', 'inverse_freq', or None")


# -----------------------------
# Model Training
# -----------------------------

def train_svm_classifier(X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: Optional[np.ndarray] = None,
                        C: float = 1.0,
                        kernel: str = 'rbf',
                        gamma: str = 'scale',
                        class_weight: Optional[Any] = 'balanced') -> Tuple[SVC, object]:
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
                      - 'balanced': automatically weights classes inversely proportional to frequency
                      - None: no class weighting
                      - dict: custom weights mapping class labels to weights
                      Example: {'class0': 1.0, 'class1': 2.0}
        
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
                          class_weight: Optional[Any] = None) -> Dict[str, Any]:
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
                      Can be 'balanced', None, or a dict mapping classes to weights
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    X_val_scaled = scaler.transform(X_val)
    y_pred = classifier.predict(X_val_scaled)
    
    unique_classes = np.unique(y_val).tolist()
    class_names = [str(c) for c in unique_classes]
    
    overall_accuracy = float(accuracy_score(y_val, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_val, y_pred))
    
    # F1 scores
    f1_macro = float(f1_score(y_val, y_pred, average='macro', zero_division=0))
    f1_weighted = float(f1_score(y_val, y_pred, average='weighted', zero_division=0))
    f1_per_class = f1_score(y_val, y_pred, labels=unique_classes, average=None, zero_division=0)
    f1_per_class_dict = {str(cls): float(score) for cls, score in zip(unique_classes, f1_per_class)}
    
    # Precision scores
    precision_macro = float(precision_score(y_val, y_pred, average='macro', zero_division=0))
    precision_weighted = float(precision_score(y_val, y_pred, average='weighted', zero_division=0))
    precision_per_class = precision_score(y_val, y_pred, labels=unique_classes, average=None, zero_division=0)
    precision_per_class_dict = {str(cls): float(score) for cls, score in zip(unique_classes, precision_per_class)}
    
    # Recall scores
    recall_macro = float(recall_score(y_val, y_pred, average='macro', zero_division=0))
    recall_weighted = float(recall_score(y_val, y_pred, average='weighted', zero_division=0))
    recall_per_class = recall_score(y_val, y_pred, labels=unique_classes, average=None, zero_division=0)
    recall_per_class_dict = {str(cls): float(score) for cls, score in zip(unique_classes, recall_per_class)}
    
    per_class_accuracy = {}
    for cls in unique_classes:
        mask = y_val == cls
        if mask.sum() > 0:
            class_acc = float(accuracy_score(y_val[mask], y_pred[mask]))
            per_class_accuracy[str(cls)] = class_acc
    
    cm = confusion_matrix(y_val, y_pred, labels=unique_classes)
    confusion_matrix_dict = {
        "labels": class_names,
        "matrix": cm.tolist()
    }
    
    report_dict = classification_report(
        y_val, y_pred, target_names=class_names, output_dict=True
    )
    
    # Class distribution
    class_dist = get_class_distribution(y_val)
    
    # Use provided parameters or fall back to classifier attributes
    class_weight_str = str(class_weight) if class_weight is not None else str(classifier.class_weight)
    model_params = {
        "C": float(C if C is not None else classifier.C),
        "kernel": str(kernel if kernel is not None else classifier.kernel),
        "gamma": str(gamma if gamma is not None else classifier.gamma),
        "class_weight": class_weight_str
    }
    
    return {
        "overall_accuracy": overall_accuracy,
        "overall_accuracy_percent": overall_accuracy * 100,
        "balanced_accuracy": balanced_acc,
        "balanced_accuracy_percent": balanced_acc * 100,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_per_class": f1_per_class_dict,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "precision_per_class": precision_per_class_dict,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "recall_per_class": recall_per_class_dict,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion_matrix_dict,
        "classification_report": report_dict,
        "model_parameters": model_params,
        "validation_samples": int(len(y_val)),
        "class_distribution": class_dist
    }


def evaluate_classifier(classifier: SVC,
                       scaler: object,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       C: Optional[float] = None,
                       kernel: Optional[str] = None,
                       gamma: Optional[Any] = None,
                       class_weight: Optional[Any] = None) -> Dict[str, Any]:
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
                      Can be 'balanced', None, or a dict mapping classes to weights
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    metrics = get_evaluation_metrics(
        classifier, scaler, X_val, y_val, C=C, kernel=kernel, gamma=gamma, class_weight=class_weight
    )
    
    print(f"\n{'='*60}")
    print("SVM CLASSIFIER EVALUATION RESULTS")
    print(f"{'='*60}")
    
    # Class distribution
    class_dist = metrics.get('class_distribution', {})
    if class_dist:
        print(f"\n0. Class Distribution (Validation Set):")
        print(f"   Total samples: {class_dist.get('total_samples', 0)}")
        print(f"   Number of classes: {class_dist.get('num_classes', 0)}")
        print(f"   Imbalance ratio: {class_dist.get('imbalance_ratio', 0):.2f}x")
        print(f"   Class counts:")
        for cls, count in sorted(class_dist.get('class_counts', {}).items()):
            print(f"     {cls}: {count}")
    
    # Overall metrics
    print(f"\n1. Overall Metrics:")
    print(f"   Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy_percent']:.2f}%)")
    print(f"   Balanced Accuracy: {metrics['balanced_accuracy']:.4f} ({metrics['balanced_accuracy_percent']:.2f}%)")
    print(f"   F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"   F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"   Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"   Precision (Weighted): {metrics['precision_weighted']:.4f}")
    print(f"   Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"   Recall (Weighted): {metrics['recall_weighted']:.4f}")
    
    # Per-class metrics
    print(f"\n2. Per-Class Metrics:")
    labels = metrics['confusion_matrix']['labels']
    print(f"   {'Class':<15} {'Accuracy':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
    print(f"   {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for cls in labels:
        acc = metrics['per_class_accuracy'].get(cls, 0.0)
        f1 = metrics['f1_per_class'].get(cls, 0.0)
        prec = metrics['precision_per_class'].get(cls, 0.0)
        rec = metrics['recall_per_class'].get(cls, 0.0)
        print(f"   {cls:<15} {acc:<12.4f} {f1:<12.4f} {prec:<12.4f} {rec:<12.4f}")
    
    # Identify underperforming classes
    print(f"\n3. Underperforming Classes (F1 < 0.5):")
    underperforming = [
        cls for cls in labels 
        if metrics['f1_per_class'].get(cls, 0.0) < 0.5
    ]
    if underperforming:
        for cls in underperforming:
            f1 = metrics['f1_per_class'].get(cls, 0.0)
            print(f"   {cls}: F1={f1:.4f}")
    else:
        print("   None")
    
    # Confusion matrix
    print(f"\n4. Confusion Matrix:")
    cm = metrics['confusion_matrix']['matrix']
    print("   Predicted ->")
    header = "   " + " ".join([f"{cls:>10}" for cls in labels])
    print(header)
    for i, cls in enumerate(labels):
        row = f"   {cls:>10} " + " ".join([f"{val:>10}" for val in cm[i]])
        print(row)
    
    # Detailed classification report
    print(f"\n5. Detailed Classification Report:")
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
                     class_weight: Optional[Any] = None,
                     base_path: str = ".") -> Path:
    """
    Save metrics JSON for a single model configuration.
    
    Args:
        metrics: Evaluation metrics dictionary
        C: C parameter value
        kernel: Kernel type
        gamma: Gamma parameter value
        class_weight: Class weight strategy or dict (optional)
        base_path: Base directory to save in
        
    Returns:
        Path to saved metrics file
    """
    results_dir = Path(base_path) / "grid_search_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create descriptive filename
    gamma_str = str(gamma).replace('.', '_')
    if class_weight is None:
        cw_str = "none"
    elif isinstance(class_weight, dict):
        cw_str = "custom"
    else:
        cw_str = str(class_weight).replace('.', '_')
    
    filename = f"metrics_C{C}_kernel{kernel}_gamma{gamma_str}_cw{cw_str}.json"
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
    
    # Display class distribution
    train_dist = get_class_distribution(y_train)
    val_dist = get_class_distribution(y_val)
    print(f"\nTraining set class distribution:")
    print(f"  Imbalance ratio: {train_dist['imbalance_ratio']:.2f}x")
    print(f"  Class counts: {train_dist['class_counts']}")
    print(f"\nValidation set class distribution:")
    print(f"  Imbalance ratio: {val_dist['imbalance_ratio']:.2f}x")
    print(f"  Class counts: {val_dist['class_counts']}")
    
    # Define parameter grid for grid search
    C_values = [0.1, 1, 10, 100]
    # C_values = [0.1]
    gamma_values = ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    # gamma_values = ['scale']
    kernel_values = ['rbf', 'linear', 'poly']
    # kernel_values = ['rbf']
    
    # Class weight strategies to test
    # Options: 'balanced', 'inverse_freq', None, or list of strategies
    class_weight_strategies = ['balanced']  # Can add: ['balanced', 'inverse_freq', None]
    # class_weight_strategies = ['balanced', 'inverse_freq']  # Uncomment to test multiple strategies

    # Create grid search results directory
    results_dir = Path("grid_search_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("GRID SEARCH: Testing all hyperparameter combinations")
    print(f"{'='*60}")
    total_combinations = len(C_values) * len(gamma_values) * len(kernel_values) * len(class_weight_strategies)
    print(f"Total combinations: {total_combinations}")
    print(f"C values: {C_values}")
    print(f"Gamma values: {gamma_values}")
    print(f"Kernel values: {kernel_values}")
    print(f"Class weight strategies: {class_weight_strategies}")
    print(f"Results will be saved to: {results_dir}")
    print(f"{'='*60}\n")
    
    best_accuracy = 0.0
    best_balanced_acc = 0.0
    best_f1 = 0.0
    best_classifier = None
    best_scaler = None
    best_metrics = None
    best_params = None
    combination_num = 0
    
    # Iterate over all parameter combinations
    for C, gamma, kernel, cw_strategy in product(C_values, gamma_values, kernel_values, class_weight_strategies):
        combination_num += 1
        
        # Skip invalid combinations (gamma not applicable for linear kernel)
        if kernel == 'linear' and gamma != 'scale':
            continue
        
        # Compute class weights if strategy is specified
        if cw_strategy is None:
            class_weight = None
            cw_display = "None"
        elif isinstance(cw_strategy, dict):
            class_weight = cw_strategy
            cw_display = str(cw_strategy)
        else:
            class_weight = compute_class_weights(y_train, strategy=cw_strategy)
            cw_display = f"{cw_strategy} (weights: {class_weight})" if class_weight else cw_strategy
        
        print(f"\n[{combination_num}] Testing: C={C}, kernel={kernel}, gamma={gamma}, class_weight={cw_strategy}")
        
        try:
            # Train classifier with current parameters
            classifier, scaler = train_svm_classifier(
                X_train, y_train, X_val, C=C, kernel=kernel, gamma=gamma, class_weight=class_weight
            )
            
            # Evaluate classifier with original parameters
            metrics = get_evaluation_metrics(
                classifier, scaler, X_val, y_val, C=C, kernel=kernel, gamma=gamma, class_weight=class_weight
            )
            accuracy = metrics['overall_accuracy']
            balanced_acc = metrics['balanced_accuracy']
            f1_weighted = metrics['f1_weighted']
            
            print(f"    Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"    Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
            print(f"    F1-Score (Weighted): {f1_weighted:.4f}")
            
            # Save metrics JSON for this model
            metrics_path = save_metrics_json(metrics, C, kernel, gamma, class_weight=class_weight)
            print(f"    Metrics saved to: {metrics_path.name}")
            
            # Track best model (using balanced accuracy as primary metric for imbalanced data)
            if balanced_acc > best_balanced_acc:
                best_accuracy = accuracy
                best_balanced_acc = balanced_acc
                best_f1 = f1_weighted
                best_classifier = classifier
                best_scaler = scaler
                best_metrics = metrics
                best_params = {
                    'C': C, 
                    'kernel': kernel, 
                    'gamma': gamma, 
                    'class_weight': class_weight,
                    'class_weight_strategy': cw_strategy
                }
                print(f"    *** NEW BEST (Balanced Acc: {balanced_acc:.4f})! ***")
        
        except Exception as e:
            print(f"    ERROR: {str(e)}")
            continue
    
    # Print summary and save best model
    print(f"\n{'='*60}")
    print("GRID SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"Best Balanced Accuracy: {best_balanced_acc:.4f} ({best_balanced_acc*100:.2f}%)")
    print(f"Best F1-Score (Weighted): {best_f1:.4f}")
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



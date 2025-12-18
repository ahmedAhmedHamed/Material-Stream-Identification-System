from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import joblib

python_dir = Path(__file__).parent
sys.path.insert(0, str(python_dir))

from classifiers.KNN import WeightedKNeighborsClassifier


_knn_classifier: Optional[object] = None
_knn_scaler: Optional[object] = None
_svm_classifier: Optional[object] = None
_svm_scaler: Optional[object] = None


def load_models(assets_dir: str) -> None:
    global _knn_classifier, _knn_scaler, _svm_classifier, _svm_scaler
    
    knn_classifier_path = os.path.join(assets_dir, "knn_classifier.joblib")
    knn_scaler_path = os.path.join(assets_dir, "knn_scaler.joblib")
    svm_classifier_path = os.path.join(assets_dir, "svm_classifier.joblib")
    svm_scaler_path = os.path.join(assets_dir, "svm_scaler.joblib")
    
    if os.path.exists(knn_classifier_path) and os.path.exists(knn_scaler_path):
        _knn_classifier = joblib.load(knn_classifier_path)
        _knn_scaler = joblib.load(knn_scaler_path)
    
    if os.path.exists(svm_classifier_path) and os.path.exists(svm_scaler_path):
        _svm_classifier = joblib.load(svm_classifier_path)
        _svm_scaler = joblib.load(svm_scaler_path)


def predict_knn(features: np.ndarray) -> Tuple[str, float]:
    if _knn_classifier is None or _knn_scaler is None:
        raise RuntimeError("KNN model not loaded")
    
    features_2d = features.reshape(1, -1)
    features_scaled = _knn_scaler.transform(features_2d)
    prediction = _knn_classifier.predict(features_scaled)[0]
    
    if hasattr(_knn_classifier, 'predict_proba'):
        probabilities = _knn_classifier.predict_proba(features_scaled)[0]
        classes = _knn_classifier.classes_
        prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
        confidence = prob_dict.get(str(prediction), 0.0)
    else:
        confidence = 1.0
    
    return str(prediction), float(confidence)


def predict_svm(features: np.ndarray) -> Tuple[str, float]:
    if _svm_classifier is None or _svm_scaler is None:
        raise RuntimeError("SVM model not loaded")
    
    features_2d = features.reshape(1, -1)
    features_scaled = _svm_scaler.transform(features_2d)
    prediction = _svm_classifier.predict(features_scaled)[0]
    
    if hasattr(_svm_classifier, 'predict_proba'):
        probabilities = _svm_classifier.predict_proba(features_scaled)[0]
        classes = _svm_classifier.classes_
        prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
        confidence = prob_dict.get(str(prediction), 0.0)
    else:
        confidence = 1.0
    
    return str(prediction), float(confidence)


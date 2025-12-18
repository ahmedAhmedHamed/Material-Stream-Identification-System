from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional
import sys
from datetime import datetime
import uuid

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from vector_extraction.feature_extractor_v2_core import extract_features
from classifiers.KNN import WeightedKNeighborsClassifier
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

KNN_MODEL_DIR = Path("classifiers/KNN_best")
SVM_MODEL_DIR = Path("classifiers/SVM_23")

_knn_classifier = None
_knn_scaler = None
_svm_classifier = None
_svm_scaler = None

MIN_CERTAINTY = 0.60
KNN_MAX_DISTANCE = 0.80
UNKNOWN_LABEL = "unknown"


class ClassificationRequest(BaseModel):
    image: str
    classifier: str


class ClassificationResponse(BaseModel):
    className: str
    confidence: float


def _register_custom_classes_for_unpickling() -> None:
    try:
        from classifiers.KNN import WeightedKNeighborsClassifier
    except Exception:
        return

    sys.modules.setdefault("__main__", sys.modules.get("__main__") or sys.modules[__name__])
    main_mod = sys.modules["__main__"]
    if not hasattr(main_mod, "WeightedKNeighborsClassifier"):
        setattr(main_mod, "WeightedKNeighborsClassifier", WeightedKNeighborsClassifier)


def _load_models():
    global _knn_classifier, _knn_scaler, _svm_classifier, _svm_scaler
    
    _register_custom_classes_for_unpickling()
    
    if KNN_MODEL_DIR.exists():
        knn_clf_path = KNN_MODEL_DIR / "knn_classifier.joblib"
        knn_scl_path = KNN_MODEL_DIR / "knn_scaler.joblib"
        if knn_clf_path.exists() and knn_scl_path.exists():
            _knn_classifier = joblib.load(knn_clf_path)
            _knn_scaler = joblib.load(knn_scl_path)
    
    if SVM_MODEL_DIR.exists():
        svm_clf_path = SVM_MODEL_DIR / "svm_classifier.joblib"
        svm_scl_path = SVM_MODEL_DIR / "svm_scaler.joblib"
        if svm_clf_path.exists() and svm_scl_path.exists():
            _svm_classifier = joblib.load(svm_clf_path)
            _svm_scaler = joblib.load(svm_scl_path)


def _base64_to_image(base64_string: str) -> np.ndarray:
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_array = np.array(image, dtype=np.uint8)
    return image_array


def _extract_features_from_image(image: np.ndarray) -> np.ndarray:
    features = extract_features(image)
    return features.reshape(1, -1)


def _save_frame(image: np.ndarray) -> None:
    frames_dir = Path("./frames")
    frames_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = frames_dir / f"frame_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
    Image.fromarray(image).save(filename)


def _predict_with_confidence(classifier, scaler, features_scaled: np.ndarray) -> tuple[str, float]:
    prediction = classifier.predict(features_scaled)[0]
    print(prediction)
    if hasattr(classifier, "n_neighbors") and classifier.n_neighbors == 1 and hasattr(classifier, "kneighbors"):
        distances, _ = classifier.kneighbors(features_scaled, n_neighbors=1, return_distance=True)
        distance = float(distances[0, 0])
        if distance > KNN_MAX_DISTANCE:
            return UNKNOWN_LABEL, 0.0
        confidence = max(0.0, 1.0 - (distance / KNN_MAX_DISTANCE))
        return str(prediction), confidence
    
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(features_scaled)[0]
        max_prob = float(np.max(probabilities))
        if max_prob < MIN_CERTAINTY:
            return UNKNOWN_LABEL, 0.0
        return str(prediction), max_prob
    
    if hasattr(classifier, "decision_function"):
        scores = np.asarray(classifier.decision_function(features_scaled))
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        max_prob = float(np.max(probs[0]))
        if max_prob < MIN_CERTAINTY:
            return UNKNOWN_LABEL, 0.0
        return str(prediction), max_prob
    
    return str(prediction), 1.0


@app.on_event("startup")
async def startup_event():
    _load_models()


@app.post("/classify", response_model=ClassificationResponse)
async def classify(request: ClassificationRequest):
    try:
        image = _base64_to_image(request.image)
        _save_frame(image)
        features = _extract_features_from_image(image)
        
        if request.classifier == "knn":
            if _knn_classifier is None or _knn_scaler is None:
                raise HTTPException(status_code=500, detail="KNN model not loaded")
            features_scaled = _knn_scaler.transform(features)
            className, confidence = _predict_with_confidence(_knn_classifier, _knn_scaler, features_scaled)
        elif request.classifier == "svm":
            if _svm_classifier is None or _svm_scaler is None:
                raise HTTPException(status_code=500, detail="SVM model not loaded")
            features_scaled = _svm_scaler.transform(features)
            className, confidence = _predict_with_confidence(_svm_classifier, _svm_scaler, features_scaled)
        else:
            raise HTTPException(status_code=400, detail="Invalid classifier type. Use 'knn' or 'svm'")
        
        return ClassificationResponse(className=className, confidence=confidence)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}


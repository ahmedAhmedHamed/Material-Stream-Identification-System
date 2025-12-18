from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple
import sys

import joblib

DATA_DIR = r"./dataset/trash"
MODEL_DIR = r"./classifiers/KNN_best"
UNKNOWN_LABEL = "unknown"
MIN_CERTAINTY = 0.60
KNN_MAX_DISTANCE = 0.80


def _ensure_repo_on_syspath(repo_root: Path) -> None:
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _iter_image_paths(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths, key=lambda p: p.as_posix().lower())


def _resolve_model_paths(model_dir: Path) -> Tuple[Path, Path]:
    svm_clf = model_dir / "svm_classifier.joblib"
    svm_scl = model_dir / "svm_scaler.joblib"
    knn_clf = model_dir / "knn_classifier.joblib"
    knn_scl = model_dir / "knn_scaler.joblib"

    if knn_clf.exists() and knn_scl.exists():
        return knn_clf, knn_scl
    if svm_clf.exists() and svm_scl.exists():
        return svm_clf, svm_scl

    raise FileNotFoundError(
        "Could not find a supported model in bestModelPath. Expected either:\n"
        f"- {knn_clf.name} + {knn_scl.name}\n"
        f"- {svm_clf.name} + {svm_scl.name}"
    )


def _load_model_and_scaler(model_dir: Path):
    classifier_path, scaler_path = _resolve_model_paths(model_dir)
    _register_custom_classes_for_unpickling()
    return joblib.load(classifier_path), joblib.load(scaler_path)


def _expected_feature_dim(scaler) -> int:
    if hasattr(scaler, "n_features_in_"):
        return int(getattr(scaler, "n_features_in_"))
    raise AttributeError("Scaler is missing n_features_in_ attribute.")


def _predict_for_paths(image_paths: Iterable[Path], classifier, scaler) -> List[str]:
    from vector_extraction.deep_feature_extractor import load_image  # pylint: disable=import-error

    preds: List[str] = []
    expected_dim = _expected_feature_dim(scaler)
    batch_size = 128
    image_paths_list = list(image_paths)
    for i in range(0, len(image_paths_list), batch_size):
        batch_paths = image_paths_list[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                images.append(load_image(str(p)))
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: could not load image '{p}': {exc}")
        if not images:
            continue
        feats = _extract_features(images, expected_dim)
        feats_scaled = scaler.transform(feats)
        batch_preds = _labels_with_unknown_threshold(
            classifier, feats_scaled, MIN_CERTAINTY, UNKNOWN_LABEL
        )
        preds.extend(batch_preds)
    return preds


def _extract_features(images, expected_dim: int):
    from vector_extraction.feature_extractor_v2_core import (  # pylint: disable=import-error
        extract_features,
    )

    feats_list = [extract_features(img) for img in images]
    import numpy as np

    feats = np.asarray(feats_list)
    if feats.ndim != 2 or feats.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dim mismatch: extracted {feats.shape} but scaler expects {expected_dim}."
        )
    return feats


def _register_custom_classes_for_unpickling() -> None:
    """
    Some saved KNN models may reference WeightedKNeighborsClassifier from __main__.
    Ensure it is available to joblib.load() in both module namespaces.
    """
    try:
        from classifiers.KNN import WeightedKNeighborsClassifier  # pylint: disable=import-error
    except Exception:  # noqa: BLE001
        return

    sys.modules.setdefault("__main__", sys.modules.get("__main__") or sys.modules[__name__])
    main_mod = sys.modules["__main__"]
    if not hasattr(main_mod, "WeightedKNeighborsClassifier"):
        setattr(main_mod, "WeightedKNeighborsClassifier", WeightedKNeighborsClassifier)


def _labels_with_unknown_threshold(classifier, X_scaled, min_certainty: float, unknown_label: str) -> List[str]:
    import numpy as np

    if getattr(classifier, "n_neighbors", None) == 1 and hasattr(classifier, "kneighbors"):
        distances, _ = classifier.kneighbors(X_scaled, n_neighbors=1, return_distance=True)
        distances = np.asarray(distances).reshape(-1)
        pred = classifier.predict(X_scaled).astype(str)
        pred = np.where(distances <= float(KNN_MAX_DISTANCE), pred, unknown_label)
        return pred.tolist()

    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(X_scaled)
        probs = np.asarray(probs)
        best = probs.max(axis=1)
        pred = classifier.predict(X_scaled).astype(str)
        pred = np.where(best >= min_certainty, pred, unknown_label)
        return pred.tolist()

    if hasattr(classifier, "decision_function"):
        scores = np.asarray(classifier.decision_function(X_scaled))
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        best = probs.max(axis=1)
        pred = classifier.predict(X_scaled).astype(str)
        pred = np.where(best >= min_certainty, pred, unknown_label)
        return pred.tolist()

    pred = classifier.predict(X_scaled).astype(str).tolist()
    return [p for p in pred]


def predict(dataFilePath: str, bestModelPath: str) -> List[str]:
    repo_root = Path(__file__).resolve().parent
    _ensure_repo_on_syspath(repo_root)

    data_dir = Path(dataFilePath).expanduser().resolve()
    model_dir = Path(bestModelPath).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"dataFilePath does not exist: {data_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"bestModelPath does not exist: {model_dir}")

    image_paths = _iter_image_paths(data_dir)
    if not image_paths:
        return []

    classifier, scaler = _load_model_and_scaler(model_dir)
    preds = _predict_for_paths(image_paths, classifier, scaler)
    return preds


def main() -> List[str]:
    if not DATA_DIR or not MODEL_DIR:
        raise ValueError("Set DATA_DIR and MODEL_DIR at the top of test.py before running.")

    preds = predict(DATA_DIR, MODEL_DIR)
    return preds


def labels_to_int(preds: List[str]) -> List[int]:
    """Map string labels to integer values.
    
    glass = 0
    paper = 1
    cardboard = 2
    plastic = 3
    metal = 4
    trash = 5
    unknown = 6
    """
    label_map = {
        "glass": 0,
        "paper": 1,
        "cardboard": 2,
        "plastic": 3,
        "metal": 4,
        "trash": 5,
        "unknown": 6,
    }
    return [label_map.get(label.lower(), 6) for label in preds]


if __name__ == "__main__":
    preds = main()
    print(labels_to_int(preds))



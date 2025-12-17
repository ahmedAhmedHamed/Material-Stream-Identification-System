from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple
import sys

import joblib

DATA_DIR = r"./dataset/trash"
MODEL_DIR = r"./classifiers/SVM_22"


def _ensure_repo_on_syspath(repo_root: Path) -> None:
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _iter_image_paths(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths, key=lambda p: p.as_posix().lower())


def _resolve_model_paths(model_dir: Path) -> Tuple[Path, Path]:
    clf = model_dir / "svm_classifier.joblib"
    scl = model_dir / "svm_scaler.joblib"
    if not clf.exists():
        raise FileNotFoundError(f"Missing classifier file: {clf}")
    if not scl.exists():
        raise FileNotFoundError(f"Missing scaler file: {scl}")
    return clf, scl


def _load_model_and_scaler(model_dir: Path):
    classifier_path, scaler_path = _resolve_model_paths(model_dir)
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
        batch_preds = classifier.predict(feats_scaled).tolist()
        preds.extend([str(x) for x in batch_preds])
    return preds


def _extract_features(images, expected_dim: int):
    if expected_dim == 1536:
        from vector_extraction.deep_feature_extractor import (  # pylint: disable=import-error
            extract_features_batch,
        )

        return extract_features_batch(images)

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
    return _predict_for_paths(image_paths, classifier, scaler)


def main() -> List[str]:
    if not DATA_DIR or not MODEL_DIR:
        raise ValueError("Set DATA_DIR and MODEL_DIR at the top of test.py before running.")

    preds = predict(DATA_DIR, MODEL_DIR)
    return preds


if __name__ == "__main__":
    preds = main()
    print(preds)



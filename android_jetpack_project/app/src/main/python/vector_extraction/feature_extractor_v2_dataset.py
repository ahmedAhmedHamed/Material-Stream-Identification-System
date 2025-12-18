from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from vector_extraction.augmentation import augment_image
from vector_extraction.class_balancing import build_class_balance_plan
from vector_extraction.feature_extractor_v2_config import DEFAULT_CONFIG
from vector_extraction.feature_extractor_v2_core import extract_features, process_image_for_features


def get_label_from_path(image_path: str) -> str:
    return Path(image_path).parent.name


def get_image_paths(root: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    return [str(p) for p in Path(root).rglob("*") if p.suffix.lower() in exts]


def analyze_class_distribution(image_paths: List[str]) -> Dict[str, List[str]]:
    class_to_paths: Dict[str, List[str]] = {}
    for p in image_paths:
        class_to_paths.setdefault(get_label_from_path(p), []).append(p)
    return class_to_paths


def process_images_parallel(image_paths: List[str], config: Dict[str, Any] | None = None) -> List[Tuple[np.ndarray, str]]:
    cfg = config or DEFAULT_CONFIG
    max_workers = int(cfg.get("max_workers", 30))
    results: List[Tuple[np.ndarray, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fut = {executor.submit(process_image_for_features, p, get_label_from_path, cfg): p for p in image_paths}
        for future in as_completed(fut):
            results.append(future.result())
    return results


def process_validation_set(class_to_val_paths: Dict[str, List[str]], config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    all_paths: List[str] = []
    for paths in class_to_val_paths.values():
        all_paths.extend(paths)
    results = process_images_parallel(all_paths, config=config)
    return _to_xy_arrays(results)


def process_training_set(
    class_to_train_paths: Dict[str, List[str]],
    augmentation_multiplier: float,
    random_state: int,
    augmentation_technique: str,
    augmentation_intensity: str,
    augmentation_groups: Tuple[str, ...] | None,
    augmentation_allowlist: Tuple[str, ...] | None,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    results: List[Tuple[np.ndarray, str]] = []
    plan = build_class_balance_plan(
        class_to_train_paths,
        random_state=random_state,
        target_multiplier=float(augmentation_multiplier),
        target="median",
    )
    for label, paths in plan.selected_paths.items():
        if not paths:
            continue
        results.extend(process_images_parallel(paths, config=config))
        aug_n = int(plan.augmented_needed.get(label, 0))
        if aug_n > 0:
            aug_imgs = _generate_augmented_images(
                paths,
                aug_n,
                random_state=random_state,
                augmentation_technique=augmentation_technique,
                augmentation_intensity=augmentation_intensity,
                augmentation_groups=augmentation_groups,
                augmentation_allowlist=augmentation_allowlist,
            )
            results.extend(_extract_augmented_parallel(aug_imgs, config=config))
    return _to_xy_arrays(results)


def build_feature_matrix(
    image_paths: List[str],
    train_output_path: str = "../features_train_v2.npz",
    val_output_path: str = "../features_val_v2.npz",
    test_size: float = 0.3,
    random_state: int = 42,
    augmentation_multiplier: float = 1.0,
    augmentation_technique: str = "random",
    augmentation_intensity: str = "moderate",
    augmentation_groups: Tuple[str, ...] | None = None,
    augmentation_allowlist: Tuple[str, ...] | None = None,
    config: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = config or DEFAULT_CONFIG
    class_to_paths = analyze_class_distribution(image_paths)
    class_to_train, class_to_val = _split_train_val(class_to_paths, test_size, random_state)
    X_val, y_val = process_validation_set(class_to_val, config=cfg)
    X_train, y_train = process_training_set(
        class_to_train,
        augmentation_multiplier,
        random_state=random_state,
        augmentation_technique=augmentation_technique,
        augmentation_intensity=augmentation_intensity,
        augmentation_groups=augmentation_groups,
        augmentation_allowlist=augmentation_allowlist,
        config=cfg,
    )
    np.savez(train_output_path, X=X_train, y=y_train)
    np.savez(val_output_path, X=X_val, y=y_val)
    return X_train, y_train, X_val, y_val


def scale_features(X_train: np.ndarray, X_val: np.ndarray | None = None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_val is None:
        return X_train_scaled, scaler
    return X_train_scaled, scaler.transform(X_val), scaler


def _split_train_val(
    class_to_paths: Dict[str, List[str]],
    test_size: float,
    random_state: int,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    class_to_train: Dict[str, List[str]] = {}
    class_to_val: Dict[str, List[str]] = {}
    for label, paths in class_to_paths.items():
        tr, va = train_test_split(paths, test_size=test_size, random_state=random_state)
        class_to_train[label] = tr
        class_to_val[label] = va
    return class_to_train, class_to_val


def _to_xy_arrays(results: List[Tuple[np.ndarray, str]]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray([f for f, _ in results], dtype=np.float32)
    y = np.asarray([lbl for _, lbl in results])
    return X, y


def _generate_augmented_images(
    image_paths: List[str],
    needed_count: int,
    *,
    random_state: int,
    augmentation_technique: str,
    augmentation_intensity: str,
    augmentation_groups: Tuple[str, ...] | None,
    augmentation_allowlist: Tuple[str, ...] | None,
) -> List[Tuple[np.ndarray, str]]:
    label = get_label_from_path(image_paths[0])
    out: List[Tuple[np.ndarray, str]] = []
    rng = random.Random(int(random_state))
    for _ in range(needed_count):
        src = rng.choice(image_paths)
        image = cv2.imread(src)
        if image is None:
            raise IOError(f"Could not read image: {src}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        out.append(
            (
                augment_image(
                    rgb,
                    technique=augmentation_technique,
                    intensity=augmentation_intensity,
                    allowed_groups=augmentation_groups,
                    allowed_augmentations=augmentation_allowlist,
                ),
                label,
            )
        )
    return out


def _extract_augmented_parallel(samples: List[Tuple[np.ndarray, str]], config: Dict[str, Any]) -> List[Tuple[np.ndarray, str]]:
    max_workers = int(config.get("max_workers", 30))
    results: List[Tuple[np.ndarray, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fut = {executor.submit(_process_aug_sample, s, config): s for s in samples}
        for future in as_completed(fut):
            results.append(future.result())
    return results


def _process_aug_sample(sample: Tuple[np.ndarray, str], config: Dict[str, Any]) -> Tuple[np.ndarray, str]:
    img, label = sample
    return extract_features(img, config=config), label



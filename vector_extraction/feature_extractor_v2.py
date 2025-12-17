"""
feature_extractor_v2.py

Public v2 entrypoints (modular classical feature extraction).
Saves `.npz` with keys `X` and `y` (same schema as v1).
"""

from __future__ import annotations

import cv2

from vector_extraction.feature_extractor_v2_config import DEFAULT_CONFIG, FEATURE_BLOCKS
from vector_extraction.feature_extractor_v2_core import extract_features, feature_report
from vector_extraction.feature_extractor_v2_dataset import (
    analyze_class_distribution,
    build_feature_matrix,
    get_image_paths,
    get_label_from_path,
    process_images_parallel,
    scale_features,
)

__all__ = [
    "DEFAULT_CONFIG",
    "FEATURE_BLOCKS",
    "extract_features",
    "feature_report",
    "analyze_class_distribution",
    "build_feature_matrix",
    "get_image_paths",
    "get_label_from_path",
    "process_images_parallel",
    "scale_features",
]


if __name__ == "__main__":
    print("Feature extractor v2 ready.")
    paths = get_image_paths("../dataset")
    if not paths:
        print("No images found in ../dataset (adjust path if needed).")
        raise SystemExit(0)
    img = cv2.cvtColor(cv2.imread(paths[0]), cv2.COLOR_BGR2RGB)
    print("Per-block dims:", feature_report(img))



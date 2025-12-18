from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np

from vector_extraction.feature_extractor_v2_config import DEFAULT_CONFIG, FEATURE_BLOCKS
from vector_extraction.features import preprocess
from vector_extraction.features.registry import build_feature_pipeline, extract_with_pipeline


def extract_features(image_rgb_u8: np.ndarray, config: Dict[str, Any] | None = None) -> np.ndarray:
    cfg = config or DEFAULT_CONFIG
    sample = preprocess.preprocess_to_sample(image_rgb_u8, cfg)
    pipeline = build_feature_pipeline(FEATURE_BLOCKS, cfg)
    return extract_with_pipeline(pipeline, sample, cfg)


def feature_report(image_rgb_u8: np.ndarray, config: Dict[str, Any] | None = None) -> Dict[str, int]:
    cfg = config or DEFAULT_CONFIG
    sample = preprocess.preprocess_to_sample(image_rgb_u8, cfg)
    pipeline = build_feature_pipeline(FEATURE_BLOCKS, cfg)
    dims: Dict[str, int] = {}
    for block in pipeline:
        dims[block.name] = int(np.asarray(block.extract(sample, cfg)).size)
    dims["total"] = int(sum(dims.values()))
    return dims


def process_image_for_features(
    image_path: str,
    get_label: Any,
    config: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, str]:
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    feats = extract_features(rgb, config=config)
    return feats, get_label(image_path)



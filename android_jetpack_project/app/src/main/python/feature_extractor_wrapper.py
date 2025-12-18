from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np

python_dir = Path(__file__).parent
sys.path.insert(0, str(python_dir))

from vector_extraction.feature_extractor_v2_core import extract_features
from vector_extraction.feature_extractor_v2_config import DEFAULT_CONFIG


def extract_features_from_array(image_array: np.ndarray) -> np.ndarray:
    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    
    if len(image_array.shape) == 3 and image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]
    
    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got shape {image_array.shape}")
    
    features = extract_features(image_array, config=DEFAULT_CONFIG)
    return features.astype(np.float32)


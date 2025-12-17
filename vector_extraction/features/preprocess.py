from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Sample:
    rgb_u8: np.ndarray
    gray_u8: np.ndarray
    hsv_u8: np.ndarray
    lab_u8: np.ndarray


def preprocess_to_sample(image_rgb_u8: np.ndarray, config: Dict) -> Sample:
    rgb = _resize_rgb(image_rgb_u8, tuple(config.get("image_size", (256, 256))))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return Sample(rgb_u8=rgb, gray_u8=gray, hsv_u8=hsv, lab_u8=lab)


def _resize_rgb(image_rgb_u8: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    if image_rgb_u8.dtype != np.uint8:
        image_rgb_u8 = np.clip(image_rgb_u8, 0, 255).astype(np.uint8)
    return cv2.resize(image_rgb_u8, size, interpolation=cv2.INTER_LINEAR)



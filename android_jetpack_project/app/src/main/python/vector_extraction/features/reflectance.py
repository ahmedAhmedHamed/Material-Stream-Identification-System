from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from vector_extraction.features._utils import fraction_above_threshold, normalize_hist


def extract_gradient_magnitude_hist(sample: Any, cfg: Dict) -> np.ndarray:
    gray = sample.gray_u8.astype(np.float32)
    bins = int(cfg.get("bins", 32))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    hist, _ = np.histogram(mag, bins=bins, range=(0.0, 255.0))
    return normalize_hist(hist)


def extract_specular_ratio(sample: Any, cfg: Dict) -> np.ndarray:
    threshold = float(cfg.get("threshold", 240.0))
    channel = str(cfg.get("channel", "v")).lower()

    if channel == "v":
        vals = sample.hsv_u8[:, :, 2]
    elif channel == "gray":
        vals = sample.gray_u8
    else:
        vals = sample.rgb_u8[:, :, 0]
    return np.asarray([fraction_above_threshold(vals, threshold)], dtype=np.float32)



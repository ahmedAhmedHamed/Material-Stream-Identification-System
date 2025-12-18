from __future__ import annotations

from typing import Tuple

import numpy as np


def normalize_hist(hist: np.ndarray) -> np.ndarray:
    hist = np.asarray(hist, dtype=np.float32)
    s = float(hist.sum())
    if s <= 0.0:
        return np.zeros_like(hist, dtype=np.float32)
    return (hist / s).astype(np.float32, copy=False)


def safe_skew(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0
    mean = float(x.mean())
    std = float(x.std())
    if std <= 1e-6:
        return 0.0
    z3 = np.mean(((x - mean) / std) ** 3)
    return float(z3)


def fraction_above_threshold(values: np.ndarray, threshold: float) -> float:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.mean(x >= threshold))


def dct2(image_gray_f32: np.ndarray) -> np.ndarray:
    import cv2

    x = np.asarray(image_gray_f32, dtype=np.float32)
    return cv2.dct(x)


def band_slices(shape: Tuple[int, int], fractions: Tuple[float, float]) -> Tuple[slice, slice, slice]:
    h, w = shape
    low_end = int(min(h, w) * fractions[0])
    mid_end = int(min(h, w) * fractions[1])
    low = (slice(0, low_end), slice(0, low_end))
    mid = (slice(low_end, mid_end), slice(low_end, mid_end))
    high = (slice(mid_end, h), slice(mid_end, w))
    return low, mid, high



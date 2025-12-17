from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from vector_extraction.features._utils import normalize_hist, safe_skew


def extract_color_moments(sample: Any, cfg: Dict) -> np.ndarray:
    space = str(cfg.get("space", "hsv")).lower()
    img = sample.hsv_u8 if space == "hsv" else sample.lab_u8
    channels = list(cfg.get("channels", [0, 1, 2]))

    feats: List[float] = []
    for ch in channels:
        vals = img[:, :, int(ch)].astype(np.float32).reshape(-1)
        feats.append(float(vals.mean()))
        feats.append(float(vals.std()))
        feats.append(safe_skew(vals))
    return np.asarray(feats, dtype=np.float32)


def extract_color_histograms(sample: Any, cfg: Dict) -> np.ndarray:
    space = str(cfg.get("space", "hsv")).lower()
    img = sample.hsv_u8 if space == "hsv" else sample.lab_u8
    bins = list(cfg.get("bins", [16, 16, 16]))
    channels = list(cfg.get("channels", [0, 1, 2]))

    parts: List[np.ndarray] = []
    for i, ch in enumerate(channels):
        b = int(bins[i]) if i < len(bins) else int(bins[-1])
        vals = img[:, :, int(ch)].reshape(-1)
        hist, _ = np.histogram(vals, bins=b, range=(0, 256))
        parts.append(normalize_hist(hist))
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)



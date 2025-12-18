from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from vector_extraction.features._utils import band_slices, dct2


def extract_dct_energy_bands(sample: Any, cfg: Dict) -> np.ndarray:
    fractions = tuple(cfg.get("fractions", (0.15, 0.40)))
    gray = sample.gray_u8.astype(np.float32) / 255.0

    coeff = dct2(gray)
    mag2 = (np.abs(coeff) ** 2).astype(np.float32)
    low_sl, mid_sl, high_sl = band_slices(mag2.shape, fractions=fractions)

    low = float(mag2[low_sl].sum())
    mid = float(mag2[mid_sl].sum())
    high = float(mag2[high_sl].sum())
    total = max(low + mid + high, 1e-12)
    return np.asarray([low / total, mid / total, high / total], dtype=np.float32)


def extract_fft_energy_bands(sample: Any, cfg: Dict) -> np.ndarray:
    fractions = tuple(cfg.get("fractions", (0.15, 0.40)))
    gray = sample.gray_u8.astype(np.float32) / 255.0

    fft = np.fft.fft2(gray)
    mag2 = (np.abs(fft) ** 2).astype(np.float32)
    mag2 = np.fft.fftshift(mag2)

    h, w = mag2.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    rmax = float(r.max()) or 1.0
    r_norm = r / rmax

    f1, f2 = float(fractions[0]), float(fractions[1])
    low = float(mag2[r_norm <= f1].sum())
    mid = float(mag2[(r_norm > f1) & (r_norm <= f2)].sum())
    high = float(mag2[r_norm > f2].sum())
    total = max(low + mid + high, 1e-12)
    return np.asarray([low / total, mid / total, high / total], dtype=np.float32)



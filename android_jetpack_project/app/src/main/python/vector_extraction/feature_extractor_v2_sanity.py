"""
Sanity checks for feature_extractor_v2.

Run:
  python -m vector_extraction.feature_extractor_v2_sanity
"""

from __future__ import annotations

import numpy as np

from vector_extraction import feature_extractor_v2 as v2


def run_basic_sanity() -> None:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    report = v2.feature_report(img)
    vec = v2.extract_features(img)
    _assert_vector_ok(vec, expected_dim=report["total"])


def _assert_vector_ok(vec: np.ndarray, expected_dim: int) -> None:
    if vec.dtype != np.float32:
        raise AssertionError(f"Expected float32, got {vec.dtype}")
    if vec.ndim != 1:
        raise AssertionError(f"Expected 1D, got shape {vec.shape}")
    if vec.size != expected_dim:
        raise AssertionError(f"Expected dim {expected_dim}, got {vec.size}")
    if not np.isfinite(vec).all():
        raise AssertionError("Vector contains NaN/Inf")


if __name__ == "__main__":
    run_basic_sanity()
    print("v2 sanity: ok")



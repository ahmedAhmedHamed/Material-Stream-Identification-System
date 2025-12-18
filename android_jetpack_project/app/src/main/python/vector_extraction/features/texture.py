from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from vector_extraction.features._utils import normalize_hist


def extract_glcm(sample: Any, cfg: Dict) -> np.ndarray:
    gray = sample.gray_u8
    levels = int(cfg.get("levels", 8))
    distances = list(cfg.get("distances", [1, 2]))
    angles = list(cfg.get("angles", [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]))
    props = list(cfg.get("properties", ["contrast", "homogeneity", "energy"]))

    quant = (gray.astype(np.uint16) * levels) // 256
    quant = np.clip(quant, 0, levels - 1).astype(np.uint8)
    glcm = graycomatrix(quant, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)

    feats: List[float] = []
    for prop in props:
        vals = graycoprops(glcm, prop)
        feats.extend(vals.flatten().tolist())
    return np.asarray(feats, dtype=np.float32)


def extract_lbp_multi_radius(sample: Any, cfg: Dict) -> np.ndarray:
    gray = sample.gray_u8
    radii: Iterable[int] = cfg.get("radii", [1, 2, 3])
    points = int(cfg.get("points", 8))
    bins = int(cfg.get("bins", 59))
    method = str(cfg.get("method", "uniform"))

    parts: List[np.ndarray] = []
    for r in radii:
        lbp = local_binary_pattern(gray, P=points, R=int(r), method=method)
        hist, _ = np.histogram(lbp, bins=bins, range=(0, bins))
        parts.append(normalize_hist(hist))
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def extract_gabor(sample: Any, cfg: Dict) -> np.ndarray:
    gray = sample.gray_u8
    orientations = int(cfg.get("orientations", 4))
    sigmas = list(cfg.get("sigmas", [2.0, 4.0]))
    lambdas = list(cfg.get("lambdas", [6.0, 10.0]))
    gamma = float(cfg.get("gamma", 0.5))
    psi = float(cfg.get("psi", 0.0))
    ksize = tuple(cfg.get("ksize", (31, 31)))

    thetas = np.linspace(0, np.pi, orientations, endpoint=False)
    feats: List[float] = []
    for sigma in sigmas:
        for lambd in lambdas:
            for theta in thetas:
                kernel = cv2.getGaborKernel(ksize=ksize, sigma=float(sigma), theta=float(theta), lambd=float(lambd), gamma=gamma, psi=psi)
                resp = cv2.filter2D(gray, cv2.CV_32F, kernel)
                feats.append(float(resp.mean()))
                feats.append(float(resp.std()))
    return np.asarray(feats, dtype=np.float32)



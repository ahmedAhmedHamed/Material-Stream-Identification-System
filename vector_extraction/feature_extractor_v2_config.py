from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from vector_extraction.features import color, frequency, reflectance, texture
from vector_extraction.features.registry import ConfiguredBlock


DEFAULT_CONFIG: Dict[str, Any] = {
    "image_size": (256, 256),
    "max_workers": 30,
    "blocks": {
        "glcm": {
            "enabled": True,
            "levels": 8,
            "distances": [1, 2],
            "angles": [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            "properties": ["contrast", "homogeneity", "energy"],
        },
        "lbp": {"enabled": True, "radii": [1, 2, 3], "points": 8, "bins": 59, "method": "uniform"},
        "gabor": {
            "enabled": True,
            "orientations": 4,
            "sigmas": [2.0, 4.0],
            "lambdas": [6.0, 10.0],
            "gamma": 0.5,
            "psi": 0.0,
            "ksize": (31, 31),
        },
        "moments_hsv": {"enabled": True, "space": "hsv", "channels": [0, 1, 2]},
        "moments_lab": {"enabled": True, "space": "lab", "channels": [0, 1, 2]},
        "hist_hsv": {"enabled": True, "space": "hsv", "channels": [0, 1, 2], "bins": [16, 16, 16]},
        "hist_lab": {"enabled": False, "space": "lab", "channels": [0, 1, 2], "bins": [16, 16, 16]},
        "grad_mag": {"enabled": True, "bins": 32},
        "specular": {"enabled": True, "threshold": 240.0, "channel": "v"},
        "dct_bands": {"enabled": True, "fractions": (0.15, 0.40)},
        "fft_bands": {"enabled": False, "fractions": (0.15, 0.40)},
    },
}


FEATURE_BLOCKS: List[ConfiguredBlock] = [
    ConfiguredBlock(name="glcm", extractor=texture.extract_glcm, config_key="glcm"),
    ConfiguredBlock(name="lbp_multi_radius", extractor=texture.extract_lbp_multi_radius, config_key="lbp"),
    ConfiguredBlock(name="gabor", extractor=texture.extract_gabor, config_key="gabor"),
    ConfiguredBlock(name="color_moments_hsv", extractor=color.extract_color_moments, config_key="moments_hsv"),
    ConfiguredBlock(name="color_moments_lab", extractor=color.extract_color_moments, config_key="moments_lab"),
    ConfiguredBlock(name="color_hist_hsv", extractor=color.extract_color_histograms, config_key="hist_hsv"),
    ConfiguredBlock(name="color_hist_lab", extractor=color.extract_color_histograms, config_key="hist_lab"),
    ConfiguredBlock(name="gradient_mag_hist", extractor=reflectance.extract_gradient_magnitude_hist, config_key="grad_mag"),
    ConfiguredBlock(name="specular_ratio", extractor=reflectance.extract_specular_ratio, config_key="specular"),
    ConfiguredBlock(name="dct_energy_bands", extractor=frequency.extract_dct_energy_bands, config_key="dct_bands"),
    ConfiguredBlock(name="fft_energy_bands", extractor=frequency.extract_fft_energy_bands, config_key="fft_bands"),
]



from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Protocol

import numpy as np


class FeatureBlock(Protocol):
    name: str

    def extract(self, sample: Any) -> np.ndarray: ...


@dataclass(frozen=True)
class ConfiguredBlock:
    name: str
    extractor: Callable[[Any, Dict[str, Any]], np.ndarray]
    config_key: str

    def extract(self, sample: Any, config: Dict[str, Any]) -> np.ndarray:
        block_cfg = _get_block_config(config, self.config_key)
        return self.extractor(sample, block_cfg)


def build_feature_pipeline(
    blocks: Iterable[ConfiguredBlock],
    config: Dict[str, Any],
) -> List[ConfiguredBlock]:
    enabled: List[ConfiguredBlock] = []
    for block in blocks:
        if _is_block_enabled(config, block.config_key):
            enabled.append(block)
    return enabled


def extract_with_pipeline(
    pipeline: Iterable[ConfiguredBlock],
    sample: Any,
    config: Dict[str, Any],
) -> np.ndarray:
    parts: List[np.ndarray] = []
    for block in pipeline:
        vec = block.extract(sample, config)
        parts.append(_as_1d_float32(vec, block.name))
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _is_block_enabled(config: Dict[str, Any], config_key: str) -> bool:
    return bool(_get_block_config(config, config_key).get("enabled", True))


def _get_block_config(config: Dict[str, Any], config_key: str) -> Dict[str, Any]:
    blocks_cfg = config.get("blocks", {})
    return dict(blocks_cfg.get(config_key, {}))


def _as_1d_float32(vec: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(vec)
    if arr.ndim != 1:
        raise ValueError(f"Feature block '{name}' returned shape {arr.shape}, expected 1D")
    return arr.astype(np.float32, copy=False)



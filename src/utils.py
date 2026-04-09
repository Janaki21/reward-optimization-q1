from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable, List

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories(paths: Iterable[str]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def sample_learning_curve_indices(n_episodes: int, n_points: int) -> List[int]:
    if n_points <= 1:
        return [n_episodes - 1]

    idx = np.linspace(0, n_episodes - 1, num=min(n_points, n_episodes), dtype=int)
    return sorted(set(idx.tolist()))


def confidence_interval_95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return (np.nan, np.nan)
    mean = float(np.mean(values))
    if n == 1:
        return (mean, mean)
    std = float(np.std(values, ddof=1))
    margin = 1.96 * (std / math.sqrt(n))
    return (mean - margin, mean + margin)


def stderr(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(n))
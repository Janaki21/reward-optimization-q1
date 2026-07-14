from pathlib import Path
from typing import Iterable


def ensure_directories(paths: Iterable[str]) -> None:
    for path in paths:
        Path(path).mkdir(
            parents=True,
            exist_ok=True,
        )
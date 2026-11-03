"""The interface every evaluated model implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Query:
    """One question for a model.

    Real models must only use `image` and `prompt`; the remaining fields describe the
    question for reference models (oracle, constant, random) and for logging.
    """

    image: np.ndarray  # (H, W) uint8 grayscale
    prompt: str  # question plus answer-format instruction
    family: str
    options: list[str] | None
    target_class: str
    region: str | None


class VQAModel(Protocol):
    name: str
    coord_format: str  # how the model writes boxes: "pixel", "norm1" or "norm1000"

    def answer(self, query: Query) -> str: ...

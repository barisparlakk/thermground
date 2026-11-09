"""Build a model from a reference-model name or a model config file (configs/models/)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

from thermground.data.hituav import Sample
from thermground.models.base import VQAModel
from thermground.models.reference import REFERENCE_MODELS, reference_model


def load_model(
    spec: str,
    samples: dict[str, Sample],
    load_image: Callable[[str], np.ndarray],
    buckets: list[str],
) -> tuple[VQAModel, dict]:
    """Returns the model and the config to log with its results."""
    if spec in REFERENCE_MODELS:
        return reference_model(spec, samples, load_image, buckets), {"name": spec}
    path = Path(spec)
    if path.suffix not in (".yaml", ".yml") or not path.exists():
        raise ValueError(f"{spec!r} is neither a reference model {REFERENCE_MODELS} nor a config")
    cfg = yaml.safe_load(path.read_text())
    if cfg["backend"] == "mlx_vlm":
        from thermground.models.mlx_vlm import MlxVLM

        model = MlxVLM(
            cfg["name"],
            cfg["model_id"],
            cfg["coord_format"],
            cfg["max_tokens"],
            cfg.get("revision"),
        )
        return model, {"config": str(path), **cfg}
    if cfg["backend"] == "hf":
        from thermground.models.hf_vlm import HFVLM

        model = HFVLM(
            cfg["name"],
            cfg["model_id"],
            cfg["coord_format"],
            cfg["max_tokens"],
            cfg.get("revision"),
            adapter=cfg.get("adapter"),
            dtype=cfg.get("dtype", "bfloat16"),
        )
        return model, {"config": str(path), **cfg}
    raise ValueError(f"unknown backend {cfg['backend']!r} in {path}")

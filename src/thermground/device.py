"""The one place that picks the compute device: cuda > mps > cpu."""

from __future__ import annotations


def get_device(prefer: str | None = None) -> str:
    """Best available device, or `prefer` if given. Without torch installed, "cpu"."""
    if prefer:
        return prefer
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

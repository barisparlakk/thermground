"""Vision-language models run with Hugging Face transformers (optional `train` extra).

Used for evaluation on CUDA, optionally with a LoRA adapter from scripts/train_lora.py.
Prompts are rendered exactly as in training (thermground.data.sft). Decoding is greedy.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from thermground.data.sft import messages
from thermground.device import get_device
from thermground.models.base import Query


def to_rgb(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image).convert("RGB")


def load_base(model_id: str, revision: str | None, dtype: str):
    """Processor and model; the model stays on the CPU until moved."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=revision, dtype=getattr(torch, dtype)
    )
    return processor, model


class HFVLM:
    def __init__(
        self,
        name: str,
        model_id: str,
        coord_format: str,
        max_tokens: dict[str, int],
        revision: str | None = None,
        adapter: str | None = None,
        dtype: str = "bfloat16",
        device: str | None = None,
    ) -> None:
        self.name = name
        self.coord_format = coord_format
        self.max_tokens = max_tokens  # {"short": ..., "point": ...}
        self.device = get_device(device)
        self.processor, model = load_base(model_id, revision, dtype)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.to(self.device).eval()

    def answer(self, query: Query) -> str:
        import torch

        text = self.processor.apply_chat_template(
            messages(query.prompt), tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[to_rgb(query.image)], return_tensors="pt")
        inputs = inputs.to(self.device)
        limit = self.max_tokens["point" if query.family == "point" else "short"]
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=limit, do_sample=False)
        new = out[0, inputs["input_ids"].shape[1] :]
        return self.processor.decode(new, skip_special_tokens=True).strip()

"""Vision-language models run with mlx-vlm on Apple silicon (optional `mlx` extra).

Decoding is greedy (temperature 0) so runs are repeatable.
"""

from __future__ import annotations

from PIL import Image

from thermground.models.base import Query


class MlxVLM:
    def __init__(
        self,
        name: str,
        model_id: str,
        coord_format: str,
        max_tokens: dict[str, int],
        revision: str | None = None,
    ) -> None:
        from mlx_vlm import load  # imported here so the package works without the extra

        self.name = name
        self.coord_format = coord_format
        self.max_tokens = max_tokens  # {"short": ..., "point": ...}
        self.model, self.processor = load(model_id, revision=revision)

    def answer(self, query: Query) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        image = Image.fromarray(query.image).convert("RGB")
        prompt = apply_chat_template(self.processor, self.model.config, query.prompt, num_images=1)
        limit = self.max_tokens["point" if query.family == "point" else "short"]
        result = generate(
            self.model,
            self.processor,
            prompt,
            image=[image],
            max_tokens=limit,
            temperature=0.0,
            verbose=False,
        )
        return result.text.strip()

"""Reference models that need no weights, used to validate the evaluation.

- oracle      answers from the annotations of the image it is shown (found by pixel hash),
              phrased as sentences so the parser is exercised too
- always_yes  yes / 7+ / one fixed box: a model that ignores the image
- always_no   no / 0 / none: always "nothing here"
- random      a random valid answer, seeded by the query itself
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np

from thermground.data.hituav import Sample
from thermground.data.qa import answer_for, count_bucket
from thermground.models.base import Query


def image_hash(image: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(image).tobytes()).hexdigest()


class OracleModel:
    name = "oracle"
    coord_format = "pixel"

    def __init__(self, samples: dict[str, Sample], load_image: Callable[[str], np.ndarray]):
        self._by_hash = {image_hash(load_image(name)): s for name, s in samples.items()}

    def answer(self, query: Query) -> str:
        sample = self._by_hash.get(image_hash(query.image))  # None for a blank image
        buckets = query.options if query.family == "count" else None
        answer, boxes = answer_for(sample, query.family, query.target_class, query.region, buckets)
        if query.family in ("presence", "region"):
            return "Yes, I can see one." if answer == "yes" else "No, there is none."
        if query.family == "count":
            return f"I count {len(boxes)} people." if boxes else "There are no people."
        if not boxes:
            return "There are no people."
        return "Here they are: " + ", ".join(
            "[" + ", ".join(f"{v:.0f}" for v in box) + "]" for box in boxes
        )


class ConstantModel:
    coord_format = "pixel"

    def __init__(self, name: str, answers: dict[str, str]):
        self.name = name
        self._answers = answers

    def answer(self, query: Query) -> str:
        return self._answers[query.family]


class RandomModel:
    name = "random"
    coord_format = "pixel"

    def answer(self, query: Query) -> str:
        # seeded by the query, so results do not depend on evaluation order
        seed = int(image_hash(query.image)[:8], 16) ^ int(
            hashlib.sha1(query.prompt.encode()).hexdigest()[:8], 16
        )
        rng = np.random.default_rng(seed)
        if query.options:
            return str(rng.choice(query.options))
        boxes = []
        for _ in range(rng.integers(0, 4)):
            x, y = rng.uniform(0, 620), rng.uniform(0, 490)
            boxes.append(f"[{x:.0f}, {y:.0f}, {x + 15:.0f}, {y + 20:.0f}]")
        return ", ".join(boxes) if boxes else "none"


def reference_model(
    name: str,
    samples: dict[str, Sample],
    load_image: Callable[[str], np.ndarray],
    buckets: list[str],
):
    if name == "oracle":
        return OracleModel(samples, load_image)
    if name == "always_yes":
        answers = {
            "presence": "Yes.",
            "region": "Yes.",
            "count": buckets[-1],
            "point": "[300, 236, 340, 276]",
        }
        return ConstantModel(name, answers)
    if name == "always_no":
        answers = {
            "presence": "No.",
            "region": "No.",
            "count": count_bucket(0, buckets),
            "point": "none",
        }
        return ConstantModel(name, answers)
    if name == "random":
        return RandomModel()
    raise ValueError(f"unknown reference model {name!r}")


REFERENCE_MODELS = ("oracle", "always_yes", "always_no", "random")

"""Turn QA items into supervised fine-tuning examples.

An example is the evaluation prompt (question plus the same answer-format instruction used
by scripts/evaluate.py) and a target answer written the way the evaluation parser reads it.
The loss is computed on the target only.

Everything here is plain Python so it can be tested without torch.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from thermground.data.qa import FAMILIES, HEIGHT, WIDTH
from thermground.eval.run import build_prompt

COORD_SCALE = {"pixel": None, "norm1": 1.0, "norm1000": 1000.0}
IGNORE = -100  # label value the loss skips


def target_text(item: dict, coord_format: str = "norm1000") -> str:
    """The answer to train on: "yes"/"no", a count bucket, or one box per line / "none"."""
    if item["family"] != "point":
        return item["answer"]
    if not item["evidence"]:
        return "none"
    scale = COORD_SCALE[coord_format]
    lines = []
    for x1, y1, x2, y2 in sorted(item["evidence"], key=lambda b: (b[1], b[0])):
        if scale is None:
            box = [x1, y1, x2, y2]
        else:
            box = [x1 * scale / WIDTH, y1 * scale / HEIGHT, x2 * scale / WIDTH, y2 * scale / HEIGHT]
        if scale == 1.0:
            lines.append("[" + ", ".join(f"{v:.3f}" for v in box) + "]")
        else:
            lines.append("[" + ", ".join(str(round(v)) for v in box) + "]")
    return "\n".join(lines)


def mix_items(items: list[dict], mix: dict[str, float], seed: int) -> list[dict]:
    """Resample families: weight 2.0 repeats every item twice, 0.3 keeps a random 30%.

    Families missing from `mix` are dropped. The result is shuffled with `seed`.
    """
    rng = np.random.default_rng(seed)
    out = []
    for family in FAMILIES:
        weight = mix.get(family, 0.0)
        fam = [i for i in items if i["family"] == family]
        whole, frac = int(weight), weight - int(weight)
        out += fam * whole
        if frac and fam:
            keep = rng.choice(len(fam), size=round(frac * len(fam)), replace=False)
            out += [fam[k] for k in sorted(keep)]
    return [out[k] for k in rng.permutation(len(out))]


def make_examples(
    items: list[dict],
    instructions: dict[str, str],
    coords_text: dict[str, str],
    coord_format: str,
) -> list[dict]:
    """{"id", "image", "family", "prompt", "target"} for each item."""
    coords = coords_text[coord_format]
    return [
        {
            "id": item["id"],
            "image": item["image"],
            "family": item["family"],
            "prompt": build_prompt(item, instructions, coords),
            "target": target_text(item, coord_format),
        }
        for item in items
    ]


def messages(prompt: str, answer: str | None = None) -> list[dict]:
    """One user turn with the image before the text, plus the answer if given."""
    chat = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if answer is not None:
        chat.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return chat


def split_chat(apply_template: Callable[..., str], prompt: str, answer: str) -> tuple[str, str]:
    """The full chat text and its trailing part after the prompt (the answer and end tokens).

    `apply_template(messages, add_generation_prompt=...)` renders a chat to text.
    """
    head = apply_template(messages(prompt), add_generation_prompt=True)
    full = apply_template(messages(prompt, answer), add_generation_prompt=False)
    if not full.startswith(head) or answer not in full[len(head) :]:
        raise ValueError("chat template does not render the answer after the prompt")
    return full, full[len(head) :]


def mask_labels(
    input_ids: list[list[int]], lengths: list[int], n_answer: list[int]
) -> list[list[int]]:
    """Labels for right-padded sequences: only the last n_answer real tokens are kept."""
    labels = []
    for ids, length, n in zip(input_ids, lengths, n_answer, strict=True):
        if not 0 < n <= length:
            raise ValueError(f"{n} answer tokens in a sequence of {length}")
        row = [IGNORE] * len(ids)
        row[length - n : length] = ids[length - n : length]
        labels.append(row)
    return labels
